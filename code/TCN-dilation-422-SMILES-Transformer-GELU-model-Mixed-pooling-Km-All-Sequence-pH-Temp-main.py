import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from sklearn.gaussian_process.kernels import RBF
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import split
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')

    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        print(f"SMILES length after split: {len(sm)}")
        if len(sm) > 218:
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)

        print(f"x_id shape: {torch.tensor(x_id).shape}")
        print(f"x_seg shape: {torch.tensor(x_seg).shape}")

        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
    trfm.eval()
    trfm = trfm

    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)

    xid = xid
    xseg = xseg

    print(f"xid shape before encoding: {xid.shape}")
    print(f"xseg shape before encoding: {xseg.shape}")

    X = trfm.encode(xid)

    print(f"SMILES embedding shape before extracting final embedding: {X.shape}")
    smiles_embedding = X

    print(f"SMILES embedding shape after extracting final embedding: {smiles_embedding.shape}")

    return smiles_embedding


def sequence_to_int_vector(sequence):
    aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    max_length = 1000
    int_vector = np.zeros(max_length, dtype=int)

    truncated_sequence = sequence[:max_length]
    for i, aa in enumerate(truncated_sequence):
        int_vector[i] = aa_to_index.get(aa, 0)

    return int_vector


class RBF(nn.Module):
    def __init__(self, centers, gamma, device=None):
        super(RBF, self).__init__()
        self.device = device or torch.device('cpu')
        self.centers = torch.reshape(torch.tensor(centers, dtype=torch.float32), [1, -1]).to(self.device)
        self.gamma = gamma

    def forward(self, x):
        x = torch.reshape(x, [-1, 1]).to(self.device)
        # print(f"RBF: x device: {x.device}, centers device: {self.centers.device}")
        return torch.exp(-self.gamma * torch.square(x - self.centers)).to(self.device)


class ConditionFloatRBF(nn.Module):
    def __init__(self, embed_dim, device=None):
        super(ConditionFloatRBF, self).__init__()
        self.device = device or torch.device('cpu')

        self.rbf_params = {'pH': (np.arange(0, 14, 0.1), 10.0),
                           'Temp': (np.arange(0, 100, 1), 10.0)
                           }
        self.condition_names = ["pH", "Temp"]

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.condition_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma, device=self.device)
            self.rbf_list.append(rbf)

            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear.to(self.device))

    def forward(self, condition):
        ph_embed = None
        temp_embed = None

        for i, name in enumerate(self.condition_names):
            if name in condition:
                x = condition[name]
                if x is not None:
                    x = x.to(self.device)
                    # print(f"Condition: {name}, x device: {x.device}")
                    rbf_x = self.rbf_list[i](x)

                    rbf_x = rbf_x.to(self.device)
                    # print(f"rbf_x device after RBF: {rbf_x.device}")

                    linear_out = self.linear_list[i](rbf_x)
                    # print(f"Linear layer output device: {linear_out.device}")

                    if name == "pH":
                        ph_embed = linear_out
                    elif name == "Temp":
                        temp_embed = linear_out

        return ph_embed, temp_embed


def extract_ph_features(ph_values, temp_values, device):
    rbf_model = ConditionFloatRBF(embed_dim=500, device=device)
    ph_embed, temp_embed = rbf_model({"pH": ph_values, "Temp": temp_values})
    return ph_embed, temp_embed


class SubstrateDataset(Dataset):
    def __init__(self, df, device):
        self.device = device
        self.smiles_embeddings = np.stack(df['Smiles'].apply(lambda x: np.array(smiles_to_vec([x]))).values)
        self.seq_vectors = np.stack(df['Sequence'].apply(sequence_to_int_vector).values)
        self.y = df['log10_Km'].astype(float).values
        self.temp = df['Temp'].values
        self.ph = df['pH'].values
        self.rbf_model = ConditionFloatRBF(embed_dim=500, device=device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq_vector = torch.tensor(self.seq_vectors[idx], dtype=torch.long).to(self.device)
        smiles_embedding = torch.tensor(self.smiles_embeddings[idx], dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y[idx], dtype=torch.float32).to(self.device)
        temp_values = torch.tensor(self.temp[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        ph_values = torch.tensor(self.ph[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        temp_embed, ph_embed = self.rbf_model({"Temp": temp_values, "pH": ph_values})
        return smiles_embedding, seq_vector, temp_embed, ph_embed, y


class TCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(TCNLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, smiles_embedding_dim, seq_fc_dim, temp_dim, ph_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super(MLPModel, self).__init__()

        set_seed(42)

        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=128)
        self.tcn1 = TCNLayer(in_channels=128, out_channels=256, kernel_size=5, dilation=4, padding=8)
        self.tcn2 = TCNLayer(in_channels=256, out_channels=256, kernel_size=5, dilation=2, padding=4)
        self.tcn3 = TCNLayer(in_channels=256, out_channels=256, kernel_size=5, dilation=2, padding=4)
        self.pool_avg = nn.AvgPool1d(kernel_size=2)
        self.pool_max = nn.MaxPool1d(kernel_size=2)
        max_length = 1000
        pooled_length = max_length // (2 * 2 * 2)
        flattened_dim = 256 * pooled_length
        self.seq_fc = nn.Linear(flattened_dim, seq_fc_dim)

        layers = []
        input_dim = smiles_embedding_dim + seq_fc_dim + temp_dim + ph_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, smiles_embedding, seq_vector, temp_embed, ph_embed):
        seq_vector = seq_vector.view(-1, 1000)
        seq_vector = self.embedding(seq_vector)
        seq_vector = seq_vector.permute(0, 2, 1)
        x_seq = self.tcn1(seq_vector)
        x_seq = self.pool_avg(x_seq)
        x_seq = self.tcn2(x_seq)
        x_seq = self.pool_max(x_seq)
        x_seq = self.tcn3(x_seq)
        x_seq = self.pool_avg(x_seq)
        x_seq = x_seq.view(x_seq.size(0), -1)
        seq_vector = self.seq_fc(x_seq)
        smiles_embedding = smiles_embedding.view(smiles_embedding.size(0), -1)
        temp_embed = temp_embed.squeeze(1)
        ph_embed = ph_embed.squeeze(1)
        combined_input = torch.cat([smiles_embedding, seq_vector, temp_embed, ph_embed], dim=1)
        output = self.mlp(combined_input)
        return output


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    train_predictions = []
    train_actuals = []

    for smiles_embedding, seq_vector, temp_embed, ph_embed, batch_y in train_loader:
        smiles_embedding, seq_vector, temp_embed, ph_embed, batch_y = smiles_embedding.to(device), seq_vector.to(
            device), temp_embed.to(device), ph_embed.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(smiles_embedding, seq_vector, temp_embed, ph_embed)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_predictions.extend(outputs.squeeze().cpu().tolist())
        train_actuals.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    train_mae = mean_absolute_error(train_actuals, train_predictions)
    train_rmse = np.sqrt(mean_squared_error(train_actuals, train_predictions))
    train_r2 = r2_score(train_actuals, train_predictions)
    train_pcc, _ = pearsonr(train_actuals, train_predictions)

    return avg_loss, train_mae, train_rmse, train_r2, train_pcc


def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_predictions = []
    val_actuals = []

    with torch.no_grad():
        for smiles_embedding, seq_vector, temp_embed, ph_embed, batch_y in val_loader:
            smiles_embedding, seq_vector, temp_embed, ph_embed, batch_y = smiles_embedding.to(device), seq_vector.to(
                device), temp_embed.to(device), ph_embed.to(device), batch_y.to(device)
            outputs = model(smiles_embedding, seq_vector, temp_embed, ph_embed)
            val_loss += criterion(outputs.squeeze(), batch_y).item()
            val_predictions.extend(outputs.squeeze().cpu().tolist())
            val_actuals.extend(batch_y.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_mae = mean_absolute_error(val_actuals, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))
    val_r2 = r2_score(val_actuals, val_predictions)
    val_pcc, _ = pearsonr(val_actuals, val_predictions)

    return avg_val_loss, val_mae, val_rmse, val_r2, val_pcc


def test(model, criterion, test_loader, device):
    model.eval()
    test_predictions = []
    test_actuals = []

    with torch.no_grad():
        for smiles_embedding, seq_vector, temp_embed, ph_embed, batch_y in test_loader:
            smiles_embedding, seq_vector, temp_embed, ph_embed, batch_y = smiles_embedding.to(device), seq_vector.to(
                device), temp_embed.to(device), ph_embed.to(device), batch_y.to(device)
            outputs = model(smiles_embedding, seq_vector, temp_embed, ph_embed)
            test_predictions.extend(outputs.squeeze().cpu().tolist())
            test_actuals.extend(batch_y.cpu().numpy())

    test_mae = mean_absolute_error(test_actuals, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
    test_r2 = r2_score(test_actuals, test_predictions)
    test_pcc, _ = pearsonr(test_actuals, test_predictions)

    return test_mae, test_rmse, test_r2, test_pcc


def main():
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv(
        '/home/lyy/lyy/Km/All_Sequence/data/final_train_data.csv')
    val_df = pd.read_csv(
        '/home/lyy/lyy/Km/All_Sequence/data/final_val_data.csv')
    test_df = pd.read_csv(
        '/home/lyy/lyy/Km/All_Sequence/data/final_test_data.csv')

    train_dataset = SubstrateDataset(train_df, device)
    val_dataset = SubstrateDataset(val_df, device)
    test_dataset = SubstrateDataset(test_df, device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    smiles_embedding_dim = 1024
    seq_fc_dim = 3500
    temp_dim = 500
    ph_dim = 500
    hidden_dims = [512, 256, 128]
    output_dim = 1
    dropout_rate = 0.3

    model = MLPModel(smiles_embedding_dim=smiles_embedding_dim, seq_fc_dim=seq_fc_dim, temp_dim=temp_dim, ph_dim=ph_dim,
                     hidden_dims=hidden_dims, output_dim=output_dim, dropout_rate=dropout_rate).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_epoch = 0
    num_epochs = 250
    log_data = []
    early_stopping_patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_mae, train_rmse, train_r2, train_pcc = train(model, criterion, optimizer, train_loader,
                                                                       device)
        val_loss, val_mae, val_rmse, val_r2, val_pcc = validate(model, criterion, val_loader, device)

        log_data.append([
            epoch + 1,
            round(train_loss, 4),
            round(train_mae, 4),
            round(train_rmse, 4),
            round(train_r2, 4),
            round(train_pcc, 4),
            round(val_loss, 4),
            round(val_mae, 4),
            round(val_rmse, 4),
            round(val_r2, 4),
            round(val_pcc, 4),
        ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       '/home/lyy/lyy/Km/All_Sequence/model/TCN-dilation4-2-2-kernel_size5-SMILES-Transformer-MLP-GELU-model-Mixed-pooling-Km-All-Sequence_pH_Temp_dim500_lr0.001_aa1000_best_model_file_model.pth')  # 保存最佳模型
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}')

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    print(f"The best model was obtained at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

    # 测试模型
    model.load_state_dict(torch.load(
        '/home/lyy/lyy/Km/All_Sequence/model/TCN-dilation4-2-2-kernel_size5-SMILES-Transformer-MLP-GELU-model-Mixed-pooling-Km-All-Sequence_pH_Temp_dim500_lr0.001_aa1000_best_model_file_model.pth'))
    test_mae, test_rmse, test_r2, test_pcc = test(model, criterion, test_loader, device)

    with open(

            '/home/lyy/lyy/Km/All_Sequence/result/TCN-dilation4-2-2-kernel_size5-SMILES-Transformer-MLP-GELU-model-Mixed-pooling-Km-All-Sequence_pH_Temp_dim500_lr0.001_aa1000_training_log.txt',
            'w') as f:
        f.write(
            'Epoch\tLoss_train\tMAE_train\tRMSE_train\tR2_train\tPCC_train\tLoss_val\tMAE_val\tRMSE_val\tR2_val\tPCC_val\tMAE_test\tRMSE_test\tR2_test\tPCC_test\n')
        for log in log_data:
            f.write('\t'.join(map(lambda x: f"{x:.4f}", log)) +
                    f"\t{test_mae:.4f}\t{test_rmse:.4f}\t{test_r2:.4f}\t{test_pcc:.4f}\n")

        f.write(f'Test MAE: {test_mae:.4f}\n')
        f.write(f'Test RMSE: {test_rmse:.4f}\n')
        f.write(f'Test R2: {test_r2:.4f}\n')
        f.write(f'Test PCC: {test_pcc:.4f}\n')


if __name__ == '__main__':
    main()
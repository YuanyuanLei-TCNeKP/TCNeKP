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
        #print(f"SMILES length after split: {len(sm)}")
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

        #print(f"x_id shape: {torch.tensor(x_id).shape}")
        #print(f"x_seg shape: {torch.tensor(x_seg).shape}")

        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
    trfm.eval()
    trfm = trfm

    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)

    xid = xid
    xseg = xseg

    #print(f"xid shape before encoding: {xid.shape}")
    #print(f"xseg shape before encoding: {xseg.shape}")

    X = trfm.encode(xid)

    #print(f"SMILES embedding shape before extracting final embedding: {X.shape}")
    smiles_embedding = X

    #print(f"SMILES embedding shape after extracting final embedding: {smiles_embedding.shape}")

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
    def __init__(self, df_kcat, df_km, device):
        self.device = device
        self.smiles_embeddings_kcat = np.stack(df_kcat['Smiles'].apply(lambda x: np.array(smiles_to_vec([x]))).values)
        self.seq_vectors_kcat = np.stack(df_kcat['Sequence'].apply(sequence_to_int_vector).values)
        self.y_kcat = df_kcat['log10_Kcat'].astype(float).values
        self.temp_kcat = df_kcat['Temp'].values
        self.ph_kcat = df_kcat['pH'].values

        self.smiles_embeddings_km = np.stack(df_km['Smiles'].apply(lambda x: np.array(smiles_to_vec([x]))).values)
        self.seq_vectors_km = np.stack(df_km['Sequence'].apply(sequence_to_int_vector).values)
        self.y_km = df_km['log10_Km'].astype(float).values
        self.temp_km = df_km['Temp'].values
        self.ph_km = df_km['pH'].values

        self.rbf_model = ConditionFloatRBF(embed_dim=500, device=device)

    def __len__(self):
        return max(len(self.y_kcat), len(self.y_km))

    def __getitem__(self, idx):
        kcat_idx = idx % len(self.y_kcat)
        km_idx = idx % len(self.y_km)

        seq_vector_kcat = torch.tensor(self.seq_vectors_kcat[kcat_idx], dtype=torch.long).to(self.device)
        smiles_embedding_kcat = torch.tensor(self.smiles_embeddings_kcat[kcat_idx], dtype=torch.float32).to(self.device)
        y_kcat = torch.tensor(self.y_kcat[kcat_idx], dtype=torch.float32).to(self.device)
        temp_values_kcat = torch.tensor(self.temp_kcat[kcat_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        ph_values_kcat = torch.tensor(self.ph_kcat[kcat_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        temp_embed_kcat, ph_embed_kcat = self.rbf_model({"Temp": temp_values_kcat, "pH": ph_values_kcat})

        seq_vector_km = torch.tensor(self.seq_vectors_km[km_idx], dtype=torch.long).to(self.device)
        smiles_embedding_km = torch.tensor(self.smiles_embeddings_km[km_idx], dtype=torch.float32).to(self.device)
        y_km = torch.tensor(self.y_km[km_idx], dtype=torch.float32).to(self.device)
        temp_values_km = torch.tensor(self.temp_km[km_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        ph_values_km = torch.tensor(self.ph_km[km_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        temp_embed_km, ph_embed_km = self.rbf_model({"Temp": temp_values_km, "pH": ph_values_km})

        return (smiles_embedding_kcat, seq_vector_kcat, temp_embed_kcat, ph_embed_kcat, y_kcat,
                smiles_embedding_km, seq_vector_km, temp_embed_km, ph_embed_km, y_km)


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


class TaskAttentionGate(nn.Module):
    def __init__(self, shared_dim, task_dim, hidden_dim=128, reduction_dim=512):
        super(TaskAttentionGate, self).__init__()

        # 添加一个处理高维度输入的降维层
        self.reduction = nn.Sequential(
            nn.Linear(shared_dim + task_dim, reduction_dim),
            nn.ReLU()
        )

        self.gate_kcat = nn.Sequential(
            nn.Linear(reduction_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.gate_km = nn.Sequential(
            nn.Linear(reduction_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, shared_features, task_features):
        # 拼接特征
        concat_features = torch.cat([shared_features, task_features], dim=1)

        # 先进行降维处理高维度输入
        reduced_features = self.reduction(concat_features)

        # 生成注意力门控
        gate_kcat = self.gate_kcat(reduced_features)
        gate_km = self.gate_km(reduced_features)

        # 应用门控
        gated_kcat = shared_features * gate_kcat.expand_as(shared_features)
        gated_km = shared_features * gate_km.expand_as(shared_features)

        return gated_kcat, gated_km


class SharedFeatureModel(nn.Module):
    def __init__(self, smiles_embedding_dim, seq_fc_dim, temp_dim, ph_dim, hidden_dims, dropout_rate=0.3):
        super(SharedFeatureModel, self).__init__()

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

        total_combined_dim = smiles_embedding_dim + seq_fc_dim + temp_dim + ph_dim
        self.shared_fc = nn.Linear(total_combined_dim, 512)
        self.shared_bn = nn.BatchNorm1d(512)
        self.shared_relu = nn.ReLU()

        # 在 SharedFeatureModel 的 __init__ 方法中
        self.task_gate = TaskAttentionGate(512, total_combined_dim)

        kcat_layers = []
        kcat_input_dim = 512
        for hidden_dim in hidden_dims:
            kcat_layers.append(nn.Linear(kcat_input_dim, hidden_dim))
            kcat_layers.append(nn.BatchNorm1d(hidden_dim))
            kcat_layers.append(nn.GELU())
            kcat_layers.append(nn.Dropout(dropout_rate))
            kcat_input_dim = hidden_dim
        kcat_layers.append(nn.Linear(kcat_input_dim, 1))
        self.kcat_mlp = nn.Sequential(*kcat_layers)

        km_layers = []
        km_input_dim = 512
        for hidden_dim in hidden_dims:
            km_layers.append(nn.Linear(km_input_dim, hidden_dim))
            km_layers.append(nn.BatchNorm1d(hidden_dim))
            km_layers.append(nn.GELU())
            km_layers.append(nn.Dropout(dropout_rate))
            km_input_dim = hidden_dim
        km_layers.append(nn.Linear(km_input_dim, 1))
        self.km_mlp = nn.Sequential(*km_layers)

    def forward(self, smiles_embedding_kcat, seq_vector_kcat, temp_embed_kcat, ph_embed_kcat,
                smiles_embedding_km, seq_vector_km, temp_embed_km, ph_embed_km):
        if smiles_embedding_kcat is not None:
            seq_vector_kcat = seq_vector_kcat.view(-1, 1000)
            seq_vector_kcat = self.embedding(seq_vector_kcat)
            seq_vector_kcat = seq_vector_kcat.permute(0, 2, 1)
            x_seq_kcat = self.tcn1(seq_vector_kcat)
            x_seq_kcat = self.pool_avg(x_seq_kcat)
            x_seq_kcat = self.tcn2(x_seq_kcat)
            x_seq_kcat = self.pool_max(x_seq_kcat)
            x_seq_kcat = self.tcn3(x_seq_kcat)
            x_seq_kcat = self.pool_avg(x_seq_kcat)
            x_seq_kcat = x_seq_kcat.view(x_seq_kcat.size(0), -1)
            seq_vector_kcat = self.seq_fc(x_seq_kcat)
            smiles_embedding_kcat = smiles_embedding_kcat.view(smiles_embedding_kcat.size(0), -1)
            temp_embed_kcat = temp_embed_kcat.squeeze(1)
            ph_embed_kcat = ph_embed_kcat.squeeze(1)
            combined_input_kcat = torch.cat([smiles_embedding_kcat, seq_vector_kcat, temp_embed_kcat, ph_embed_kcat],
                                            dim=1)
            print("combined_input_kcat shape:", combined_input_kcat.shape)
            shared_features_kcat = self.shared_fc(combined_input_kcat)
            print("shared_features_kcat shape:", shared_features_kcat.shape)
            shared_features_kcat = self.shared_bn(shared_features_kcat)
            shared_features_kcat = self.shared_relu(shared_features_kcat)

            gated_kcat, _ = self.task_gate(shared_features_kcat, combined_input_kcat)
            kcat_output = self.kcat_mlp(gated_kcat)
        else:
            kcat_output = None

        if smiles_embedding_km is not None:
            seq_vector_km = seq_vector_km.view(-1, 1000)
            seq_vector_km = self.embedding(seq_vector_km)
            seq_vector_km = seq_vector_km.permute(0, 2, 1)
            x_seq_km = self.tcn1(seq_vector_km)
            x_seq_km = self.pool_avg(x_seq_km)
            x_seq_km = self.tcn2(x_seq_km)
            x_seq_km = self.pool_max(x_seq_km)
            x_seq_km = self.tcn3(x_seq_km)
            x_seq_km = self.pool_avg(x_seq_km)
            x_seq_km = x_seq_km.view(x_seq_km.size(0), -1)
            seq_vector_km = self.seq_fc(x_seq_km)
            smiles_embedding_km = smiles_embedding_km.view(smiles_embedding_km.size(0), -1)
            temp_embed_km = temp_embed_km.squeeze(1)
            ph_embed_km = ph_embed_km.squeeze(1)
            combined_input_km = torch.cat([smiles_embedding_km, seq_vector_km, temp_embed_km, ph_embed_km], dim=1)
            shared_features_km = self.shared_fc(combined_input_km)
            shared_features_km = self.shared_bn(shared_features_km)
            shared_features_km = self.shared_relu(shared_features_km)

            _, gated_km = self.task_gate(shared_features_km, combined_input_km)
            km_output = self.km_mlp(gated_km)
        else:
            km_output = None

        return kcat_output, km_output


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    train_predictions_kcat = []
    train_actuals_kcat = []
    train_predictions_km = []
    train_actuals_km = []

    for (smiles_embedding_kcat, seq_vector_kcat, temp_embed_kcat, ph_embed_kcat, y_kcat,
         smiles_embedding_km, seq_vector_km, temp_embed_km, ph_embed_km, y_km) in train_loader:
        optimizer.zero_grad()

        kcat_output, km_output = model(smiles_embedding_kcat, seq_vector_kcat, temp_embed_kcat, ph_embed_kcat,
                                       smiles_embedding_km, seq_vector_km, temp_embed_km, ph_embed_km)

        loss = 0
        if kcat_output is not None and y_kcat is not None:
            loss_kcat = criterion(kcat_output.squeeze(), y_kcat)
            loss += loss_kcat
            train_predictions_kcat.extend(kcat_output.squeeze().cpu().tolist())
            train_actuals_kcat.extend(y_kcat.cpu().numpy())

        if km_output is not None and y_km is not None:
            loss_km = criterion(km_output.squeeze(), y_km)
            loss += loss_km
            train_predictions_km.extend(km_output.squeeze().cpu().tolist())
            train_actuals_km.extend(y_km.cpu().numpy())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    if train_actuals_kcat:
        train_mae_kcat = mean_absolute_error(train_actuals_kcat, train_predictions_kcat)
        train_rmse_kcat = np.sqrt(mean_squared_error(train_actuals_kcat, train_predictions_kcat))
        train_r2_kcat = r2_score(train_actuals_kcat, train_predictions_kcat)
        train_pcc_kcat, _ = pearsonr(train_actuals_kcat, train_predictions_kcat)
    else:
        train_mae_kcat, train_rmse_kcat, train_r2_kcat, train_pcc_kcat = 0, 0, 0, 0

    if train_actuals_km:
        train_mae_km = mean_absolute_error(train_actuals_km, train_predictions_km)
        train_rmse_km = np.sqrt(mean_squared_error(train_actuals_km, train_predictions_km))
        train_r2_km = r2_score(train_actuals_km, train_predictions_km)
        train_pcc_km, _ = pearsonr(train_actuals_km, train_predictions_km)
    else:
        train_mae_km, train_rmse_km, train_r2_km, train_pcc_km = 0, 0, 0, 0

    return avg_loss, train_mae_kcat, train_rmse_kcat, train_r2_kcat, train_pcc_kcat, train_mae_km, train_rmse_km, train_r2_km, train_pcc_km


def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_predictions_kcat = []
    val_actuals_kcat = []
    val_predictions_km = []
    val_actuals_km = []

    with torch.no_grad():
        for (smiles_embedding_kcat, seq_vector_kcat, temp_embed_kcat, ph_embed_kcat, y_kcat,
             smiles_embedding_km, seq_vector_km, temp_embed_km, ph_embed_km, y_km) in val_loader:
            kcat_output, km_output = model(smiles_embedding_kcat, seq_vector_kcat, temp_embed_kcat, ph_embed_kcat,
                                           smiles_embedding_km, seq_vector_km, temp_embed_km, ph_embed_km)

            if kcat_output is not None and y_kcat is not None:
                loss_kcat = criterion(kcat_output.squeeze(), y_kcat)
                val_loss += loss_kcat.item()
                val_predictions_kcat.extend(kcat_output.squeeze().cpu().tolist())
                val_actuals_kcat.extend(y_kcat.cpu().numpy())

            if km_output is not None and y_km is not None:
                loss_km = criterion(km_output.squeeze(), y_km)
                val_loss += loss_km.item()
                val_predictions_km.extend(km_output.squeeze().cpu().tolist())
                val_actuals_km.extend(y_km.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)

    if val_actuals_kcat:
        val_mae_kcat = mean_absolute_error(val_actuals_kcat, val_predictions_kcat)
        val_rmse_kcat = np.sqrt(mean_squared_error(val_actuals_kcat, val_predictions_kcat))
        val_r2_kcat = r2_score(val_actuals_kcat, val_predictions_kcat)
        val_pcc_kcat, _ = pearsonr(val_actuals_kcat, val_predictions_kcat)
    else:
        val_mae_kcat, val_rmse_kcat, val_r2_kcat, val_pcc_kcat = 0, 0, 0, 0

    if val_actuals_km:
        val_mae_km = mean_absolute_error(val_actuals_km, val_predictions_km)
        val_rmse_km = np.sqrt(mean_squared_error(val_actuals_km, val_predictions_km))
        val_r2_km = r2_score(val_actuals_km, val_predictions_km)
        val_pcc_km, _ = pearsonr(val_actuals_km, val_predictions_km)
    else:
        val_mae_km, val_rmse_km, val_r2_km, val_pcc_km = 0, 0, 0, 0

    return avg_val_loss, val_mae_kcat, val_rmse_kcat, val_r2_kcat, val_pcc_kcat, val_mae_km, val_rmse_km, val_r2_km, val_pcc_km


def test(model, criterion, test_loader, device):
    model.eval()
    test_predictions_kcat = []
    test_actuals_kcat = []
    test_predictions_km = []
    test_actuals_km = []

    with torch.no_grad():
        for (smiles_embedding_kcat, seq_vector_kcat, temp_embed_kcat, ph_embed_kcat, y_kcat,
             smiles_embedding_km, seq_vector_km, temp_embed_km, ph_embed_km, y_km) in test_loader:
            kcat_output, km_output = model(smiles_embedding_kcat, seq_vector_kcat, temp_embed_kcat, ph_embed_kcat,
                                           smiles_embedding_km, seq_vector_km, temp_embed_km, ph_embed_km)

            if kcat_output is not None and y_kcat is not None:
                test_predictions_kcat.extend(kcat_output.squeeze().cpu().tolist())
                test_actuals_kcat.extend(y_kcat.cpu().numpy())

            if km_output is not None and y_km is not None:
                test_predictions_km.extend(km_output.squeeze().cpu().tolist())
                test_actuals_km.extend(y_km.cpu().numpy())

    if test_actuals_kcat:
        test_mae_kcat = mean_absolute_error(test_actuals_kcat, test_predictions_kcat)
        test_rmse_kcat = np.sqrt(mean_squared_error(test_actuals_kcat, test_predictions_kcat))
        test_r2_kcat = r2_score(test_actuals_kcat, test_predictions_kcat)
        test_pcc_kcat, _ = pearsonr(test_actuals_kcat, test_predictions_kcat)
    else:
        test_mae_kcat, test_rmse_kcat, test_r2_kcat, test_pcc_kcat = 0, 0, 0, 0

    if test_actuals_km:
        test_mae_km = mean_absolute_error(test_actuals_km, test_predictions_km)
        test_rmse_km = np.sqrt(mean_squared_error(test_actuals_km, test_predictions_km))
        test_r2_km = r2_score(test_actuals_km, test_predictions_km)
        test_pcc_km, _ = pearsonr(test_actuals_km, test_predictions_km)
    else:
        test_mae_km, test_rmse_km, test_r2_km, test_pcc_km = 0, 0, 0, 0

    return test_mae_kcat, test_rmse_kcat, test_r2_kcat, test_pcc_kcat, test_mae_km, test_rmse_km, test_r2_km, test_pcc_km


def main():
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df_kcat = pd.read_csv(
        '/home/lyy/lyy/Kcat-Km-integrated-model/data/Kcat/Kcat-pseudo-sequence_final_train_data4.csv')
    val_df_kcat = pd.read_csv(
        '/home/lyy/lyy/Kcat-Km-integrated-model/data/Kcat/Kcat-pseudo-sequence_final_val_data4.csv')
    test_df_kcat = pd.read_csv(
        '/home/lyy/lyy/Kcat-Km-integrated-model/data/Kcat/Kcat-pseudo-sequence_final_test_data4.csv')

    train_df_km = pd.read_csv(
        '/home/lyy/lyy/Kcat-Km-integrated-model/data/Km/Km-pseudo-sequence_final_train_data4.csv')
    val_df_km = pd.read_csv(
        '/home/lyy/lyy/Kcat-Km-integrated-model/data/Km/Km-pseudo-sequence_final_val_data4.csv')
    test_df_km = pd.read_csv(
        '/home/lyy/lyy/Kcat-Km-integrated-model/data/Km/Km-pseudo-sequence_final_test_data4.csv')

    train_dataset = SubstrateDataset(train_df_kcat, train_df_km, device)
    val_dataset = SubstrateDataset(val_df_kcat, val_df_km, device)
    test_dataset = SubstrateDataset(test_df_kcat, test_df_km, device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    smiles_embedding_dim = 1024
    seq_fc_dim = 3500
    temp_dim = 500
    ph_dim = 500
    hidden_dims = [512, 256, 128]
    dropout_rate = 0.3

    model = SharedFeatureModel(
        smiles_embedding_dim=smiles_embedding_dim,
        seq_fc_dim=seq_fc_dim,
        temp_dim=temp_dim,
        ph_dim=ph_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_epoch = 0
    num_epochs = 250
    log_data = []
    early_stopping_patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_mae_kcat, train_rmse_kcat, train_r2_kcat, train_pcc_kcat, train_mae_km, train_rmse_km, train_r2_km, train_pcc_km = train(
            model, criterion, optimizer, train_loader, device)
        val_loss, val_mae_kcat, val_rmse_kcat, val_r2_kcat, val_pcc_kcat, val_mae_km, val_rmse_km, val_r2_km, val_pcc_km = validate(
            model, criterion, val_loader, device)

        log_data.append([
            epoch + 1,
            round(train_loss, 4),
            round(train_mae_kcat, 4),
            round(train_rmse_kcat, 4),
            round(train_r2_kcat, 4),
            round(train_pcc_kcat, 4),
            round(train_mae_km, 4),
            round(train_rmse_km, 4),
            round(train_r2_km, 4),
            round(train_pcc_km, 4),
            round(val_loss, 4),
            round(val_mae_kcat, 4),
            round(val_rmse_kcat, 4),
            round(val_r2_kcat, 4),
            round(val_pcc_kcat, 4),
            round(val_mae_km, 4),
            round(val_rmse_km, 4),
            round(val_r2_km, 4),
            round(val_pcc_km, 4),
        ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       '/home/lyy/lyy/Kcat-Km-integrated-model/model/TCN-dilation4-2-2-kernel_size5-SMILES-Transformer-MLP-GELU-integrated-share-TaskAttentionGate-GELU-model-Mixed-pooling-Kcat-Km-full-Sequence_pH_Temp_repeat4_dim500_lr0.001_aa1000_best_model_file_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train MAE (Kcat): {train_mae_kcat:.4f}, Val MAE (Kcat): {val_mae_kcat:.4f}, '
              f'Train MAE (Km): {train_mae_km:.4f}, Val MAE (Km): {val_mae_km:.4f}')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    print(f"The best model was obtained at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

    # 测试模型
    model.load_state_dict(torch.load(
        '/home/lyy/lyy/Kcat-Km-integrated-model/model/TCN-dilation4-2-2-kernel_size5-SMILES-Transformer-MLP-GELU-integrated-share-TaskAttentionGate-GELU-model-Mixed-pooling-Kcat-Km-full-Sequence_pH_Temp_repeat4_dim500_lr0.001_aa1000_best_model_file_model.pth'))
    test_mae_kcat, test_rmse_kcat, test_r2_kcat, test_pcc_kcat, test_mae_km, test_rmse_km, test_r2_km, test_pcc_km = test(
        model, criterion, test_loader, device)

    with open(
            '/home/lyy/lyy/Kcat-Km-integrated-model/result/TCN-dilation4-2-2-kernel_size5-SMILES-Transformer-MLP-GELU-integrated-share-TaskAttentionGate-GELU-model-Mixed-pooling-Kcat-Km-full-Sequence_pH_Temp_repeat4_dim500_lr0.001_aa1000_training_log.txt',
            'w') as f:
        f.write(
            'Epoch\tLoss_train\tMAE_train_Kcat\tRMSE_train_Kcat\tR2_train_Kcat\tPCC_train_Kcat\tMAE_train_Km\tRMSE_train_Km\tR2_train_Km\tPCC_train_Km\tLoss_val\tMAE_val_Kcat\tRMSE_val_Kcat\tR2_val_Kcat\tPCC_val_Kcat\tMAE_val_Km\tRMSE_val_Km\tR2_val_Km\tPCC_val_Km\tMAE_test_Kcat\tRMSE_test_Kcat\tR2_test_Kcat\tPCC_test_Kcat\tMAE_test_Km\tRMSE_test_Km\tR2_test_Km\tPCC_test_Km\n')
        for log in log_data:
            f.write('\t'.join(map(lambda x: f"{x:.4f}", log)) +
                    f"\t{test_mae_kcat:.4f}\t{test_rmse_kcat:.4f}\t{test_r2_kcat:.4f}\t{test_pcc_kcat:.4f}\t{test_mae_km:.4f}\t{test_rmse_km:.4f}\t{test_r2_km:.4f}\t{test_pcc_km:.4f}\n")

        f.write(f'Test MAE (Kcat): {test_mae_kcat:.4f}\n')
        f.write(f'Test RMSE (Kcat): {test_rmse_kcat:.4f}\n')
        f.write(f'Test R2 (Kcat): {test_r2_kcat:.4f}\n')
        f.write(f'Test PCC (Kcat): {test_pcc_kcat:.4f}\n')
        f.write(f'Test MAE (Km): {test_mae_km:.4f}\n')
        f.write(f'Test RMSE (Km): {test_rmse_km:.4f}\n')
        f.write(f'Test R2 (Km): {test_r2_km:.4f}\n')
        f.write(f'Test PCC (Km): {test_pcc_km:.4f}\n')


if __name__ == '__main__':
    main()
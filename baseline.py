import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        self.node_feat = torch.tensor(data['node_feat'], dtype=torch.float32)
        self.edge_index = torch.tensor(data['edge_index'], dtype=torch.float32)
        self.node_config_feat = torch.tensor(data['node_config_feat'], dtype=torch.float32)
        self.config_runtime = torch.tensor(data['config_runtime'], dtype=torch.float32)
        return self.node_feat, self.edge_index, self.node_config_feat, self.config_runtime
    
DATA_ROOT = Path('data')

print('Loading Train data...')
ndf_files_train = list((DATA_ROOT / 'npz_all/npz/layout/nlp/default/train').iterdir())
nrd_files_train = list((DATA_ROOT / 'npz_all/npz/layout/nlp/random/train').iterdir())
xdf_files_train = list((DATA_ROOT / 'npz_all/npz/layout/xla/default/train').iterdir())
xrd_files_train = list((DATA_ROOT / 'npz_all/npz/layout/xla/random/train').iterdir())

ndf_nps_train = [np.load(f) for f in ndf_files_train]
nrd_nps_train = [np.load(f) for f in nrd_files_train]
xdf_nps_train = [np.load(f) for f in xdf_files_train]
xrd_nps_train = [np.load(f) for f in xrd_files_train]

train_dataset = GraphDataset(ndf_nps_train+nrd_nps_train+xdf_nps_train+xrd_nps_train)

import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# Graph Neural Network Layer
class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, edge_index):
        # Aggregate information
        col = edge_index[1]
        col = col.long()  # 确保索引是 int64 类型
        col = col.unsqueeze(1)

        out = self.linear(x)
        out = F.relu(out)
        out = torch.scatter_add(out, 0, col, x)  # Aggregation step
        return out

# LSTM for Config Features
class ConfigFeatureNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(ConfigFeatureNet, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the output of the last time step
        return out

# Combined Model
class CombinedModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, config_feat_dim, hidden_dim, num_layers):
        super(CombinedModel, self).__init__()
        self.gnn = GNNLayer(node_feat_dim, hidden_dim)
        self.config_net = ConfigFeatureNet(config_feat_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(2 * hidden_dim, 1)  # Final prediction layer

    def forward(self, node_feat, edge_index, config_feat):
        # Process node features and edge index with GNN
        gnn_out = self.gnn(node_feat, edge_index)
        gnn_out = torch.mean(gnn_out, dim=0)  # Global Average Pooling
        gnn_out = gnn_out.unsqueeze(1)

        # Process config features with LSTM
        config_out = self.config_net(config_feat)

        # Combine the outputs
        combined_out = torch.cat([gnn_out, config_out], dim=-1)

        # Final prediction
        out = self.fc(combined_out)
        return out

# Model instantiation with dummy dimensions
node_feat_dim = 140
edge_feat_dim = 2  # edge_index has 2 columns (source and destination nodes)
config_feat_dim = 18
hidden_dim = 64
num_layers = 2

model = CombinedModel(node_feat_dim, edge_feat_dim, config_feat_dim, hidden_dim, num_layers)


# model = GraphModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(10):
    for node_feat, edge_index, node_config_feat, config_runtime in train_dataset:
        optimizer.zero_grad()
        output = model(node_feat, edge_index, node_config_feat)
        loss = criterion(output, config_runtime)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

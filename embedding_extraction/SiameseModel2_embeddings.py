import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold 
import argparse
import random
import pickle

split = 2

class InternalProcessing(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(InternalProcessing, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        pre_process = x.clone()
        x = self.linear(x)
        x = self.relu(x)
        post_process = x.clone()
        x = self.dropout(x)
        return x, pre_process, post_process

class SiameseNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, dropout_rate):
        super(SiameseNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.internal_process = InternalProcessing(hidden_size, hidden_size, dropout_rate)
        self.fc2 = nn.Linear(hidden_size * 2, 1)
        self.output = nn.Sigmoid()

    def forward_one(self, x):
        pre_fc1 = x.clone()
        x = self.fc1(x)
        pre_attention = x.clone()
        post_fc1 = x.clone()
        post_attention = x.clone()
        x, pre_internal, post_internal = self.internal_process(x)
        return x, (pre_fc1, post_fc1), pre_attention, post_attention, pre_internal, post_internal
    
    def forward(self, reactant, product):
        reactant_out, r_fc1, r_pre_att, r_post_att, r_pre_internal, r_post_internal = self.forward_one(reactant)
        product_out, p_fc1, p_pre_att, p_post_att, p_pre_internal, p_post_internal = self.forward_one(product)
        combined = torch.cat((reactant_out, product_out), dim=1)

        pre_final_fc = combined.clone()
        output = self.fc2(combined)
        post_final_fc = output.clone()
        final_output = self.output(output)
        
        return final_output, r_fc1, p_fc1, (r_pre_att, r_post_att, r_pre_internal, r_post_internal), (p_pre_att, p_post_att, p_pre_internal, p_post_internal), pre_final_fc, post_final_fc

class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X  # tuple pairs
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x1, x2 = self.X[index]
        y = self.y[index]
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x1, x2, y

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from torch.utils.data import DataLoader

base_path = '/hpc/group/rekerlab/cem125/EnsembleSiameseProdrugs'

# import test dataset
test_dataset_bt = pd.read_csv(base_path + '/data/prodrug_test_set.csv')

# calculate and save Morgan fingerprints
substrate_mols = [Chem.MolFromSmiles(s) for s in test_dataset_bt.Reac]
substrate_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in substrate_mols]
test_dataset_bt["Reac_Molecule"] = substrate_mols
test_dataset_bt["Reac_Fingerprint"] = substrate_fps
product_mols = [Chem.MolFromSmiles(s) for s in test_dataset_bt.Prod]
product_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in product_mols]
test_dataset_bt["Prod_Molecule"] = product_mols
test_dataset_bt["Prod_Fingerprint"] = product_fps

# split X and y data for test set
X_test = list(zip(test_dataset_bt['Reac_Fingerprint'], test_dataset_bt['Prod_Fingerprint']))
y_test = test_dataset_bt['Related'].values

# initialize test dataset and data loader
test_dataset = SiameseDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# import train dataset
train_dataset_bt = pd.read_csv(base_path + '/data/train_set2.csv')

# calculate and save Morgan fingerprints
substrate_mols = [Chem.MolFromSmiles(s) for s in train_dataset_bt.Reac]
substrate_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in substrate_mols]
train_dataset_bt["Reac_Molecule"] = substrate_mols
train_dataset_bt["Reac_Fingerprint"] = substrate_fps
product_mols = [Chem.MolFromSmiles(s) for s in train_dataset_bt.Prod]
product_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in product_mols]
train_dataset_bt["Prod_Molecule"] = product_mols
train_dataset_bt["Prod_Fingerprint"] = product_fps

# split X and y data for train set
X = list(zip(train_dataset_bt['Reac_Fingerprint'], train_dataset_bt['Prod_Fingerprint']))
y = train_dataset_bt['Related'].values

# specify parameters for current train set
batch_size = 16
dropout_rate = 0.25
hidden_size = 256
lr = 0.01
num_epochs = 10

# initialize train dataset and data loader
train_dataset = SiameseDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# initialize model, loss, optimizer
model = SiameseNN(input_dim=2048, hidden_size=hidden_size, output_size=128, dropout_rate=dropout_rate)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train model and initialize latent representations
intermediate_representations = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for reac_fp, prod_fp, labels in train_loader:
        reac_fp, prod_fp, labels = reac_fp.to(device), prod_fp.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs, r_fc1, p_fc1, r_intermediate, p_intermediate, pre_final_fc, post_final_fc = model(reac_fp, prod_fp)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        intermediate_representations.append({
                    'r_pre_fc1': r_fc1[0].cpu().detach().numpy(),
                    'r_post_fc1': r_fc1[1].cpu().detach().numpy(),
                    'p_pre_fc1': p_fc1[0].cpu().detach().numpy(),
                    'p_post_fc1': p_fc1[1].cpu().detach().numpy(),
                    'reactant_pre_attention': r_intermediate[0].detach().cpu(),
                    'reactant_post_attention': r_intermediate[1].detach().cpu(),
                    'product_pre_attention': p_intermediate[0].detach().cpu(),
                    'product_post_attention': p_intermediate[1].detach().cpu(),
                    'r_pre_internal': r_intermediate[2].detach().cpu().numpy(),
                    'r_post_internal': r_intermediate[3].detach().cpu().numpy(),
                    'p_pre_internal': p_intermediate[2].detach().cpu().numpy(),
                    'p_post_internal': p_intermediate[3].detach().cpu().numpy(),
                    'pre_final_fc': pre_final_fc.detach().cpu().numpy(),
                    'post_final_fc': post_final_fc.detach().cpu().numpy()
                })

        total_loss += loss.item()

# make predictions to test set and save latent representations
model.eval()
predictions = []
embeddings = []
with torch.no_grad():
    for reac_fp, prod_fp, labels in test_loader:
        reac_fp, prod_fp, labels = reac_fp.to(device), prod_fp.to(device), labels.to(device)
        optimizer.zero_grad()
        probabilities, r_fc1, p_fc1, r_intermediate, p_intermediate, pre_final_fc, post_final_fc  = model(reac_fp, prod_fp)
        predicted = (probabilities > 0.5).float()
        predicted_probs = probabilities.numpy()
        predictions.extend(predicted_probs)
        embeddings.append({
                    'r_pre_fc1': r_fc1[0].cpu().detach().numpy(),
                    'r_post_fc1': r_fc1[1].cpu().detach().numpy(),
                    'p_pre_fc1': p_fc1[0].cpu().detach().numpy(),
                    'p_post_fc1': p_fc1[1].cpu().detach().numpy(),
                    'reactant_pre_attention': r_intermediate[0].detach().cpu(),
                    'reactant_post_attention': r_intermediate[1].detach().cpu(),
                    'product_pre_attention': p_intermediate[0].detach().cpu(),
                    'product_post_attention': p_intermediate[1].detach().cpu(),
                    'r_pre_internal': r_intermediate[2].detach().cpu().numpy(),
                    'r_post_internal': r_intermediate[3].detach().cpu().numpy(),
                    'p_pre_internal': p_intermediate[2].detach().cpu().numpy(),
                    'p_post_internal': p_intermediate[3].detach().cpu().numpy(),
                    'pre_final_fc': pre_final_fc.detach().cpu().numpy(),
                    'post_final_fc': post_final_fc.detach().cpu().numpy()
                })
predictions = np.array(predictions)
with open(base_path + '/results/ProdrugEmbeddings2.pickle', 'wb') as handle:
  pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

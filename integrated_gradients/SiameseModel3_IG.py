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


split = 3

class InternalProcessing(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(InternalProcessing, self).__init__()
        self.process = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.process(x)

class SiameseNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, attention_size, dropout_rate):
        super(SiameseNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )
        self.internal_process = InternalProcessing(hidden_size, hidden_size, dropout_rate)
        self.fc2 = nn.Linear(hidden_size * 2, 1)
        self.output = nn.Sigmoid()


    def forward_one(self, x):
        x = self.fc1(x)
        x = self.internal_process(x)
        return x

    def forward(self, reactant, product):
        reactant_out = self.forward_one(reactant)
        product_out = self.forward_one(product)
        combined = torch.cat((reactant_out, product_out), dim=1)
        output = self.fc2(combined)
        return self.output(output)
    
    def get_logits(self, reactant, product):
        reactant_out = self.forward_one(reactant)
        product_out = self.forward_one(product)
        combined = torch.cat((reactant_out, product_out), dim=1)
        logits = self.fc2(combined)
        return logits

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

def integrated_gradients(model, reac_fp, prod_fp, steps=50):
    model.eval()
    
    reac_base = torch.zeros_like(reac_fp)
    prod_base = torch.zeros_like(prod_fp)
    
    accum_reac_grads = torch.zeros_like(reac_fp)
    accum_prod_grads = torch.zeros_like(prod_fp)

    for i in range(steps + 1):
        alpha = i / steps
        reac_step = (reac_base + alpha * (reac_fp - reac_base)).detach().requires_grad_(True)
        prod_step = (prod_base + alpha * (prod_fp - prod_base)).detach().requires_grad_(True)
        
        probabilities = model(reac_step, prod_step)
        
        model.zero_grad()
        probabilities.backward(torch.ones_like(probabilities))
        
        accum_reac_grads += reac_step.grad
        accum_prod_grads += prod_step.grad
        
    reac_ig = (reac_fp - reac_base) * (accum_reac_grads / (steps + 1))
    prod_ig = (prod_fp - prod_base) * (accum_prod_grads / (steps + 1))
    
    return reac_ig.detach().cpu().numpy(), prod_ig.detach().cpu().numpy()

def set_seed(seed_value):
    """Setting seed here for reproducibility!"""
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
test_dataset_bt = pd.read_csv(base_path + '/integrated_gradients/competing_pathway_examples.csv')

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
train_dataset_bt = pd.read_csv(base_path + '/data/train_set3.csv')

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
dropout_rate = 0.1
hidden_size = 512
lr = 0.01
num_epochs = 10

# initialize model, loss, optimizer
train_dataset = SiameseDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# train model
model = SiameseNN(input_dim=2048, hidden_size=hidden_size, output_size=128, dropout_rate=dropout_rate)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for reac_fp, prod_fp, labels in train_loader:
        reac_fp, prod_fp, labels = reac_fp.to(device), prod_fp.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(reac_fp, prod_fp)
        output = output.squeeze(1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# make predictions to test set and save integrated gradients
model.eval()
predictions = []
reac_gradients = []
prod_gradients = []
all_reac_ig, all_prod_ig = [], []

for reac_fp, prod_fp, labels in test_loader:
    reac_fp, prod_fp = reac_fp.to(device).float(), prod_fp.to(device).float()
    reac_fp.requires_grad_(True)
    prod_fp.requires_grad_(True)
    
    probabilities = model(reac_fp, prod_fp)
    logits = model.get_logits(reac_fp, prod_fp)
    model.zero_grad()
    probabilities.backward(torch.ones_like(probabilities))

    reac_grad = reac_fp.grad.detach().abs().cpu().numpy()
    prod_grad = prod_fp.grad.detach().abs().cpu().numpy()
    
    predictions.extend(probabilities.detach().cpu().numpy())
    reac_gradients.append(reac_grad)
    prod_gradients.append(prod_grad)

    reac_fp.grad.zero_()
    prod_fp.grad.zero_()
    reac_ig_batch, prod_ig_batch = integrated_gradients(model, reac_fp, prod_fp, steps=50)
    input_indices = np.where(reac_fp > 0)[0]

    all_reac_ig.append(reac_ig_batch)
    all_prod_ig.append(prod_ig_batch)

predictions = np.array(predictions).flatten()
reac_grads_final = np.concatenate(all_reac_ig, axis=0)
prod_grads_final = np.concatenate(all_prod_ig, axis=0)
reac_df = pd.DataFrame(reac_grads_final).add_prefix('reac_bit_')
prod_df = pd.DataFrame(prod_grads_final).add_prefix('prod_bit_')

df_wide = pd.concat([reac_df, prod_df], axis=1)
df_wide.to_csv(base_path + '/integrated_gradients/gradients_{}.csv'.format(split))

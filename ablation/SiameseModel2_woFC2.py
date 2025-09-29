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

split = 2

class Attention(nn.Module):
    def __init__(self, feature_dim, attention_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )

    def forward(self, features):
        scores = self.attention_weights(features)
        scores = scores.squeeze(1)
        weights = F.softmax(scores, dim=0)
        weights = weights.unsqueeze(1)
        weighted = features * weights.expand_as(features)
        return weighted

class InternalProcessing(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(InternalProcessing, self).__init__()
        self.process = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.process(x)

class SiameseNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, attention_size, dropout_rate):
        super(SiameseNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc2 = nn.Linear(hidden_size * 2, 1)
        self.output = nn.Sigmoid()


    def forward_one(self, x):
        x = self.fc1(x)
        return x

    def forward(self, reactant, product):
        reactant_out = self.forward_one(reactant)
        product_out = self.forward_one(product)
        combined = torch.cat((reactant_out, product_out), dim=1)
        output = self.fc2(combined)
        return self.output(output)

class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X  
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

base_path = '/hpc/group/rekerlab/cem125/FinalMetabClassifier'

# import test dataset
test_name =  prospective_prodrug_set #'test_set{}'.format(split) # 'prodrug_test_set', 'prospective_prodrug_set'
test_dataset_bt = pd.read_csv(base_path + '/data/{}.csv'.format(test_name))

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
train_dataset_bt = pd.read_csv(base_path + '/reps/random_both_dataset_under1_rep1_new.csv')

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
attention_size =  128
batch_size = 32
dropout_rate = 0.1
hidden_size = 256
lr = 0.1
num_epochs = 20

# initialize model, loss, optimizer
train_dataset = SiameseDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# initialize model, loss, optimizer
model = SiameseNN(input_dim=2048, hidden_size=hidden_size, output_size=128, attention_size=attention_size,dropout_rate=dropout_rate)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train model
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

# make predictions to test set
model.eval()
predictions = []
embeddings = []
with torch.no_grad():
    for reac_fp, prod_fp, labels in test_loader:
        reac_fp, prod_fp, labels = reac_fp.to(device), prod_fp.to(device), labels.to(device)
        optimizer.zero_grad()

        probabilities = model(reac_fp, prod_fp)
        predicted = (probabilities > 0.5).float()
        predicted_probs = probabilities.numpy()
        predictions.extend(predicted_probs)
predictions = np.array(predictions)
pd.DataFrame(predictions).to_csv(base_path + '/results/ablation/siamese_{}_preds_woFC2.csv'.format(test_name.split('_')[0],split))
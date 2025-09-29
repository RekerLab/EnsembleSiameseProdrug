import argparse

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold 
import math
import json
from sklearn import metrics
from scipy import stats as stats
import random
from random import sample
from rdkit.Chem import AllChem, rdChemReactions, BRICS, Descriptors
from numpy.core.arrayprint import printoptions
from collections import defaultdict
import re

from lightgbm import LGBMClassifier as lgb
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import gc

split = 3

def model_builder(x_dat, ext, modelType):
  '''
  Function to fit model on current train set according
  to parameters determined in optimization, then make
  predictions to test set.

  Returns test set class predictions (0/1) and probabilities.
  '''
  if modelType == 'RF':
    model = RandomForestClassifier(max_depth = 50, max_features = 'sqrt', n_estimators = 500, random_state=42)
  if modelType == 'DT':
    model = DecisionTreeClassifier(max_depth = 100, max_features = 'sqrt', criterion = 'gini', random_state=42)
  if modelType == 'GB':
    model = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 5, n_estimators = 500, random_state=42)
  if modelType == 'XGB':
    model = xgb.XGBClassifier(booster = 'gbtree',eta = 0.5, max_depth = 5, n_estimators = 500, random_state=42)
  if modelType == 'SVM_Lin':
    model = svm.SVC(kernel = 'linear', probability = True, C = 0.1, gamma = 'scale', random_state=42)
  if modelType == 'SVM_RBF':
    model = svm.SVC(kernel = 'rbf', probability = True, C = 10, gamma = 'scale', random_state=42)
  if modelType == 'Log':
    model = LogisticRegression(solver = 'sag', C = 0.001, penalty = 'l2', random_state=42)
  if modelType == 'kNN':
    model = KNeighborsClassifier(metric = 'minkowski', n_neighbors = 9, weights = 'uniform')
  if modelType == 'MLP':
    model = MLPClassifier(activation = 'relu', alpha = 0.05, hidden_layer_sizes = (100,), learning_rate = 'constant', solver = 'adam', random_state=42)

  import warnings
  warnings.filterwarnings(action='once')

  model.fit(np.vstack(x_dat.Fingerprint.to_numpy()),x_dat.Related) # fit model on pair training 
  y_pred = model.predict(np.vstack(ext.Fingerprint.to_numpy()))
  y_prob = model.predict_proba(np.vstack(ext.Fingerprint.to_numpy()))[:, 1]
  
  return y_pred,y_prob

base_path = '/hpc/group/rekerlab/cem125/EnsembleSiameseProdrugs'

# import test dataset
test_name = 'test_set{}'.format(split) # 'prodrug_test_set', 'prospective_prodrug_set'
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
test_curated = test_dataset_bt[['Reac_Fingerprint', 'Prod_Fingerprint', 'Related']]
test_curated['Fingerprint'] = test_curated.Reac_Fingerprint.combine(test_curated.Prod_Fingerprint, np.append)
test_curated.reset_index(inplace = True)

# import train dataset
train_dataset = pd.read_csv(base_path + '/data/train_set{}.csv'.format(split))

# calculate and save Morgan fingerprints
substrate_mols = [Chem.MolFromSmiles(s) for s in train_dataset.Reac]
substrate_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in substrate_mols]
train_dataset["Reac_Molecule"] = substrate_mols
train_dataset["Reac_Fingerprint"] = substrate_fps
product_mols = [Chem.MolFromSmiles(s) for s in train_dataset.Prod]
product_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in product_mols]
train_dataset["Prod_Molecule"] = product_mols
train_dataset["Prod_Fingerprint"] = product_fps
train_curated = train_dataset[['Reac_Fingerprint', 'Prod_Fingerprint', 'Related']]
train_curated['Fingerprint'] = train_curated.Reac_Fingerprint.combine(train_curated.Prod_Fingerprint, np.append)
train_curated.reset_index(inplace = True)

# train all baseline models and make predictions to test set
model_archs = ['RF','DT','GB','XGB','SVM_RBF','SVM_Lin','Log','kNN','MLP']
model_preds = []
for mod in model_archs:
  preds, probs = model_builder(train_curated,test_curated,mod)
  model_preds.append(probs)

# save test set predictions
model_preds =  pd.DataFrame(model_preds)
model_preds.insert(loc=0, column='Model', value=model_archs)
model_preds = model_preds.T
model_preds.to_csv(base_path + '/results/benchmarking/{}_preds_{}.csv'.format(test_name.split('_')[0],split))
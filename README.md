# Ensemble Siamese Neural Networks for Prodrug Activation Prediction
This study details the development of an ensemble Siamese neural network model for predicing prodrug activation. Our model accurately predicts FDA-approved prodrugs and shows particularly remarkable performance in predicting non-traditional biotransformations, prodrug activations not captured by established metabolic reaction rules, and when predicting activation of prodrugs with lower chemical structural similarity to their API. A prospective analysis confirmed the model's ability to rank prodrug candidates by their observed release profile, establishing this tool as a generalizable resource for rational prodrug design.
<img width="1401" height="409" alt="GitHubAbstract" src="https://github.com/user-attachments/assets/9d5ecc7e-5848-4302-9487-8a27adf1183c" />

<br />

## Folder descriptions
### ablation
This folder contains code used to train and evaluate modified versions of the three SiameseNN models in the ensemble for an ablation study. Ablation study model architectures are as follows: 

<img width="758" height="346" alt="image" src="https://github.com/user-attachments/assets/a040c13b-b9c0-41c6-9bd3-04dd4bc857e8" />

### baseline_models
This folder contains code used to train and evaluate baseline ensemble models. Baseline machine learning models include:
* k-Nearest Neighbors
* Decision Tree
* Random Forest 
* Gradient Boosting
* XGBoost
* Support Vector Machine (Linear Kernel)
* Support Vector Machine (Radial Basis Function Kernel)
* Logistic Regression
* Multilayer Perceptron

### data
This folder contains data files used to train and evaluate models:
* Three train sets of endogenous and xenobiotic metabolic reactions used to train each of the three SiameseNNs in the ensemble model
* Three test sets corresponding to to each train set and SiameseNN
* An external test set of FDA-approved prodrugs

### embedding_extraction
This folder contains code used to extract latent embeddings representing the prodrug test set from each of the three SiameseNN models in the ensemble.
Due to the large file size of extracted embeddings, these are stored on Zenodo: [10.5281/zenodo.17237868](https://doi.org/10.5281/zenodo.17237868)

### integrated_gradients
This folder contains code used to calculate integrated gradients for a set of prodrugs with proven or theoretical competing metabolic pathways for each of the three SiameseNN models in the ensemble. This folder also contains gradients calculated by each model for this set of prodrugs.

### model
This folder contains code used to train and evaluate each of the three SiameseNN models in the ensemble.

### results
This folder contains data files detailing predictions made by the:
* SiameseNN and modified models 
* Baseline models
* Existing meatbolism prediction tools

<br />

## License
The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions.

# CS7641 Assignment 3: Unsupervised Learning and Dimensionality Reduction
Kirsten Odendaal (15/07/24)

REPO URL: https://github.com/kodendaal/cs_7641_a3.git

Please clone this git to a local project folder if you would like to replicate the experiments in the assignment

```git clone https://github.com/kodendaal/cs_7641_a3.git```

Requirements:
----
This file contains all the necessary packages for this project.

* Ensure that Python 3.11 is installed
* Ensure that pip and git are installed
* Includes modified pyperch repo for neural network assessments
* Running ```pip install -r requirements.txt``` will install all the packages in your project's environment


Datasets:
----
The two datasets used are the NASA Near Earth Objects and Wine Classification. They are can be found in the associated 'datasets' folder. They can also be directly downloaded from their source on Kaggle:

* https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification
* https://www.kaggle.com/datasets/yasserh/wine-quality-dataset



Files:
----

The clustering and dimensionality reduction evaluation file is contained in a jupyter notebook: ```main.ipynb```. 

The Neural Network evaluation comparisons file is contained in a jupyter notebook: ```main_nn.ipynb```. 

Helper evaluation functions are contained in a python file: ```utils.py```.


----
The general running structure is as follows;

Clustering and Dimensionality Reduction analysis:
1. Select problem you would like to evaluate and load in selection cell. 
2. Peform preliminary clustering analysis: k-Means and GMM for 10 seeded initialization 
3. Peform preliminary dimensionality reduction analysis: PCA, ICA, RP 
 * Calculate corrleation matrix for reduced feature sets
4. Re-evalaute clustering analysis on reduced dataset
5. Visualize comparison summary - plot the projected data for the first three features/components of PCA, ICA, RP

Neural Network:
1. Select dataset you would like to evaluate and load in selection cell. 
    * Perform new gridsearch evaluations (1st)
    * Indicate whether you would like to load in previous pickle files in outputdir folder
    * Plot the evalauted Learning curves
2. Load and split data into training and testing set. (Perform data standardization)
3. Manually set hyperparameter ranges and evaluate grid-search (store resulting models)
4. Using best params - Re-evaluate on multi-seed and append statistics. Evaluate final model on test set.
5. Visualize neural network learning curves
6. Visualize time complexity plots for each dimensionality reduction methods


Write-up:
----
Overleaf read-only link: https://www.overleaf.com/read/qzwdwzsncrpz#9be9dc

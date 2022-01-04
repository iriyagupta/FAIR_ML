# Fairness without Demographics through Adversarially Reweighted Learning
## Abstract from Report

This project aims to apply the concept of Adversarial Reweighted Learning (ARL) as a tool to improve the fairness of a Machine Learning (ML) model when protected features such as race or gender are not known or cannot be used. In fairness research, these protected features are used to produce fairer models, however in the “real world” this data is sometimes prohibited from use, or not available. This project strives to produce a method that is more realistically helpful for producing fair results in the “real world” by creating a fair model without the use of protected features. In this project, we reproduce the model using the same methodology as the original paper, and attempt to reproduce the paper’s results. Our results, while following the same general trend of the paper’s, do not show some of the same improvements in metrics for protected groups as the paper produces.

## Structure of Code
The two main jupyter notebooks contain the flow for loading the data and training the models. Running these files top to bottom will reproduce training results and metrics. The files are:
 - `main_compas.ipynb` containing the code to run the models on the COMPAS dataset
 - `main_law_school.ipynb` containing the code to run the models on the LSAC dataset


The files for parsing each dataset are located in
 - `data_utils`
    -  `compas_input.py` parses data for the COMPAS dataset
    -  `law_school_input.py` parses data for the LSAC dataset


The files including the csv data are located in
 - `data`
    -  `compas` includes test and train files for COMPAS dataset, as well as the raw csv data, the file containing mean and stddev data, and the vocabulary for categorical features
    -  `law_school` includes test and train files for LSAC dataset, as well as the raw csv data and the file containing mean and stddev data

The files for training the baseline and ARL models are located in
 - `train_models`


The files for creating the models are located in
 - `models`

The report writeup is:
 - `E4040.2021Fall.FAIR.report.rg3332.krh2154.al4213.pdf`

## Main Notebooks
At the top of the main notebooks are the hyperparameter settings for the models according to the paper specifications, where provided. The notebooks read in the data, stratify the testing data by group, create the models and train them, and produce AUC metrics.

## Directory Structure
```./
├── E4040.2021Fall.FAIR.report.rg3332.krh2154.al4213.pdf
├── README.md
├── data
│   ├── compas
│   │   ├── compas-scores-two-years.csv
│   │   ├── mean_std.json
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── vocabulary.json
│   └── law_school
│       ├── lsac.csv
│       ├── mean_std.json
│       ├── test.csv
│       └── train.csv
├── data_utils
│   ├── compas_input.py
│   └── law_school_input.py
├── images
│   └── results.png
├── main_compas.ipynb
├── main_law_school.ipynb
├── models
│   └── model_definitions.py
├── requirements.txt
└── train_models
    ├── train_ARL_model.py
    └── train_baseline_model.py
```

# Talent-Job Match Classifier

## Problem

The goal here is to create a lightweight search & ranking component for talent profiles and job profiles using Python and machine learning. There are two main goals:

* **Goal 1:** Create a machine learning model which takes a **talent** and **job** profile as input and returns a label (`true` if talent and job _**match**_, else `false`) and a score (a float, for ranking purposes only).

* **Goal 2:** Write a search & ranking component using the provided template (see _**search.py**_) with the machine learning model you created in the first part of this task.

## Project structure
```
talent_job_matcher/
│
├── data/
│   ├── processed/
|   |   └── job_data.json
|   |   └── talent_data.json
│   └── raw/
│       └── data.json
│
├── models/
│   └── 2024-02-28_00-16-16_DecisionTree.pkl
│   └── 2024-02-28_00-16-16_LogisticRegression.pkl
│   └── 2024-02-28_00-16-16_RandomForest.pkl
│   └── 2024-02-28_00-16-16_SVC.pkl
│   └── 2024-02-28_00-16-16_XGBoost.pkl
│
├── src/
│   ├── data_preparation.py
│   ├── model.py
│   └── utils.py
│
├── README.md
├── data_analysis.ipynb
├── inference.ipynb
├── requirements.txt
├── search.py
└── train.ipynb
```

## Setup
All dependencies are mentioned in the `requirements.txt` file. 

This file may be used to create a virtual environment using `$ conda create --name <env> --file requirements.txt` where `<env>` is the name of your new virtual environment. This uses `conda` as this project was created using `conda`.

## Data Analysis
Features from the raw data are analysed. You can find the complete analysis with suitable plots in the [Analysis](data_analysis.ipynb) notebook.

## Data Preparation/Prepropcessing
One of the important things in this project is how the data is preprocessed (see the last section for more details). Each categorical feature is converted to its numerical counterpart using reasonable logic. Code for this logic can be found in the [data_preparation.py](src/data_preparation.py) script.

## Training
Five common classifier models are used to train on the data:
* [Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
* [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/python/index.html)

K-Fold cross-validation is performed on the data with each model to compare them. 

**NOTES**:
* XGBoost is not recommended for this problem. The reason is overfitting. XGBoost is a very powerful model, but can easily overfit to simpler/smaller datasets. Here, XGBoost overfits and shows 100% accuracy. This is why it is not recommended.
* It is much better to use Decision Trees (or Random Forests which are advanced Decision Trees). They showed cross-validation accuracy of **~99.75%**.
* Logistic Regression and SVC are linear and simpler models which are not able to perform as good as the others (cross-validation accuracy **~71%-75%**).

The complete training code with plots can be found in the [train](train.ipynb) notebook.

## Inference
Here the aim is to show that the code solves both the goals (see above) of this project.

The Search class in the [search.py](search.py) script contains two methods `match()` and `match_bulk()` which are used during inference to show that the classifier algorithm works for single talent-job profile pair as well as on combination of such pairs from a list of talent profiles and job profiles.

The complete code for inference can be found in the [inference](inference.ipynb) notebook.

## Why this approach for the problem?

* The [data](data/raw/data.json) contains nested dictionary structures involving strings, lists, integers, and further dictionaries. Although the data is overall simple, the structure of the data is not really directly usable as input to a machine learning algorithm. So each categorical feature is processed to convert them logically to numerical features.
* After analysing and visualising the data, it can be understood that rule-specific ML approaches like Decision Trees and Random Forests could be good choices here. As said before, linear models like LogisticRegression and SVC give satisfactory performance but not as good as the rule-based models.

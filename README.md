



# Health-Insurance-Prediction |  A Machine Learning Based Prediction
Health Insurance Prediction using multiple ML algorithms

![590-5901121_lovely-professional-university-logo-hd-png-download](https://user-images.githubusercontent.com/62024355/120755302-6ee99700-c52b-11eb-95b8-075edac041ed.png)


__CA-3: LPU | CAP776 D2117__


![Pyhon 3.4](https://img.shields.io/badge/ide-Jupyter_notebook-blue.svg) ![Python](https://img.shields.io/badge/Language-Python-brightgreen.svg)  ![Frontend](https://img.shields.io/badge/Frontend-Bootstrap-purple.svg)  ![Frontend](https://img.shields.io/badge/Libraries-Streamlit-purple.svg)    ![Bootstrap](https://img.shields.io/badge/BaseEnvironment-AnacondaPrompt-brown.svg)   ![Bootstrap](https://img.shields.io/badge/Deployment-Github-yellow.svg)   ![Bootstrap](https://img.shields.io/badge/Debugging-LocalHost-blue.svg)  


## Table of Content
  * [Problem statment / Why this topic?](#Problem-statment)
  * [Flow Chart / Archeticture](#Flow-chart)
  * [Directory Tree](#directory-tree)
  * [Quick start](#Quick-start)
  * [Screenshots](#screenshots)
  * [Technical Aspect](#technical-aspect)
  * [Team](#team)
  * [License](#license)
  
  
  
![image](https://user-images.githubusercontent.com/62024355/196232532-c5dc622a-37d1-438d-a51f-2452068f096a.png)

  ## About project
A simple yet challenging project, to anticipate whether the insurance will be claimed or not. The complexity arises due to the fact that the dataset has fewer samples, & is slightly imbalanced. In this we challenge how we can overcome these obstacles & build a good predictive model to classify them!

Objective: 
•	Understand the Dataset & cleanup (if required).
•	Build classification model to predict weather the insurance will be claimed or not.
•	Also fine-tune the hyperparameters & compare the evaluation metrics of vaious classification algorithms.


## Flow chart


## Directory Tree


## Quick start
**Importing libraries:** 
```python
import os
import math
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn import tree
from scipy.stats import randint
from scipy.stats import loguniform
from IPython.display import display

from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from scikitplot.metrics import plot_roc_curve as auc_roc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
f1_score, roc_auc_score, roc_curve, precision_score, recall_score

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,6]

import warnings 
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 50)

```

## Screenshots
![image](https://user-images.githubusercontent.com/62024355/196595872-a0efc345-a7c0-44b2-974a-2ce0bed6c8be.png)
![image](https://user-images.githubusercontent.com/62024355/196595920-2073e25b-485c-4934-8404-50f0ced76a6a.png)
![image](https://user-images.githubusercontent.com/62024355/196595966-1ee03f6a-4d12-44fd-a7f2-22f177544b49.png)


## Technical asspects
Stractegic Plan of Action:
We aim to solve the problem statement by creating a plan of action, Here are some of the necessary steps:
1.	Data Exploration
2.	Exploratory Data Analysis (EDA)
3.	Data Pre-processing
4.	Data Manipulation
5.	Feature Selection/Extraction
6.	Predictive Modelling
7.	Project Outcomes & Conclusion


## Team


## License
[![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)](http://www.apache.org/licenses/LICENSE-2.0e)

Copyright 2021 Karan Mehra | Meenka Kalsi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

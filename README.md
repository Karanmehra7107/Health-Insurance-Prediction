



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

![Flowchart](https://user-images.githubusercontent.com/62024355/196760839-2551c8ca-9ac9-414f-a7a0-182ad46beef6.png)


## Directory Tree
```
├── insurance-claim-prediction-top-ml-models.ipynb
├── readme.md
├── insurance2.csv
└── requirements.txt


```


## Quick start
  
**Step-1:** Download the files in the repository.<br>
**Step-2:** Get into the downloaded folder, open command prompt in that directory and install all the dependencies using following command<br>
```python
pip install -r requirements.txt
```



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


1.	Exploratory Data Analysis (EDA)
                Letting first analyze the distribution of the target variable
 
![image](https://user-images.githubusercontent.com/62024355/196596644-6b36bf65-211d-40ea-a106-eff806888093.png)

2.	Data Pre-processing 
Removal of any Duplicate rows (if any)
 
![image](https://user-images.githubusercontent.com/62024355/196596667-680804d8-6025-4378-a1a4-2f4860aca101.png)

3.	Data Manipulation
                 Splitting the data intro training & testing sets
![image](https://user-images.githubusercontent.com/62024355/196596680-929c0811-1cc2-42eb-a510-79e447514090.png)

 
4.	Feature Selection/Extraction
Checking the correlation
 
![image](https://user-images.githubusercontent.com/62024355/196596699-1beae18e-4227-4aa7-8d9c-1c4e8fd37e34.png)

5.	Calculate the VIFs to remove multi co-linearity
![image](https://user-images.githubusercontent.com/62024355/196596732-177596a1-80e8-4497-bccb-65209feb4095.png)

 
6.	Predictive Modelling
 
![image](https://user-images.githubusercontent.com/62024355/196596769-bc28c0b3-90b0-4e72-9ebe-a593051ff7cb.png)
![image](https://user-images.githubusercontent.com/62024355/196596790-d054011b-d567-4163-9730-b73c7678ed0e.png)
![image](https://user-images.githubusercontent.com/62024355/196596806-6d32d854-9ce8-4e01-b3db-85a93addb9a7.png)
![image](https://user-images.githubusercontent.com/62024355/196596817-5bce0e62-1d1c-45e5-b1ff-b5707297b68e.png)
![image](https://user-images.githubusercontent.com/62024355/196596831-42f13a92-bbf7-47c0-8dee-caed6f10dd72.png)
![image](https://user-images.githubusercontent.com/62024355/196596846-5327527c-2551-476e-b2dd-f80125aeae10.png)
![image](https://user-images.githubusercontent.com/62024355/196596860-8254dfc8-e3dc-4f0b-90ba-6211b145aa18.png)
![image](https://user-images.githubusercontent.com/62024355/196596881-2520ba77-497c-45c7-8bbe-35bdbbec650c.png)
![image](https://user-images.githubusercontent.com/62024355/196596893-a15c9497-035a-4442-b76e-9ffe73cd4817.png)

 

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


__Models with their Accuracy of Prediction__

Disease | Type of Model | Accuracy
--- | --- | ---
Health care insurance | 1. Logistic Regression | Accuracy = 81.2% F1 Score = 81.3%
--- | --- | ---
Health care insurance | 2. Decisoin Tree Classfier: | Accuracy = 88.7% F1 Score = 88.7%

Health care insurance | 3. Random Forest Classfier: | Accuracy = 90.60000000000001% F1 Score = 90.60000000000001%

Health care insurance | 4. Naive Bayes Classfier: | Accuracy = 68.8% F1 Score = 68.7%

Health care insurance | 5. Support Vector Machine Classfier: | Accuracy = 84.39999999999999% F1 Score = 84.39999999999999%

Health care insurance | 6. K-Nearest Neighbours Classfier: | Accuracy = 83.6% F1 Score = 83.6%

Health care insurance | 7. Gradient Boosting Classfier: | Accuracy = 92.60000000000001% F1 Score = 92.60000000000001%

Health care insurance | 8. Extreme Gradient Boosting Classfier: | Accuracy = 93.8% F1 Score = 93.7%



## Team
[Karan Mehra (Data modeling, model integration, Front-end)](https://karanmehra7107.github.io/My-Portfolio/index.html) 
<br> [Menka Kalsi (Exploratory Data cleaning, Data gathering)](https://github.com/MenkaKalsi) 


__Special thanks to:__ 
<br> Dr. Ram Kumar (Assoicate professr)  Programming with Python | Machime Leanirng. <br>
Dr. Ashok (Head of school) Programming with Python | Machime Leanirng.

## License
[![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)](http://www.apache.org/licenses/LICENSE-2.0e)

Copyright 2022 Karan Mehra | Meenka Kalsi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

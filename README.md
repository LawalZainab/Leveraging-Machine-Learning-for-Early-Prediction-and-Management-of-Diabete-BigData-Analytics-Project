# Leveraging Machine Learning for Early Prediction and Management of Diabetes -  Big Data Analytics Project 

### Project Overview

Diabetes is a chronic metabolic disorder characterized by elevated blood sugar levels, resulting from insufficient insulin production or ineffective utilization. This disorder is known to be one of the most common chronic diseases affecting people around the globe. 
This research explores the application of machine learning in predicting diabetes, focusing on enhancing early detection and management. The outcomes aim to empower healthcare professionals with actionable insights, contributing to proactive healthcare strategies and improved patient outcomes. 


### Research Questions
- The main aim of this research is to propose a multiclass classification methodology for the prediction of diabetes.
  
This research focuses on two datasets from the laboratory of Medical City Hospital (the Specializes Center for Endocrinology and Diabetes-Al-Kindy Teaching Hospital) and another dataset from Vanderbilt, which is based on a study of rural African Americans in 
         Virginia.
1.  To employ different supervised multiclass classification algorithms that would accurately predict the onset of diabetes based on the predictor risk factors.
2.  What features are most influential in developing robust machine learning models for diabetes risk assessment?
3.  What machine learning technique is optimal to create a predictive model for diabetes?
4.  Evaluate and compare the developed system and performance of each algorithm.
5.  Comparing which dataset accurately predicts diabetes.

   
### Theme: 
Supervised Learning - Classification Algorithm

### Data Sources:
 This research paper mainly uses two datasets:
 
1- The Diabetes Dataset of 1000 Iraqi patients, acquired from the laboratory of Medical City Hospital and (the Specializes Center for Endocrinology and Diabetes-Al-Kindy Teaching Hospital). 

 [Diabetes Dataset - Mendeley Data](https://data.mendeley.com/datasets/wj9rwkp9c2/1)
 
2- The Diabetes Dataset from Vanderbilt, which is based on a study of rural African Americans in Virginia.

 https://data.world/informatics-edu/diabetes-prediction   Diabetes.csv
[[Diabetes Dataset - Mendeley Data](https://data.mendeley.com/datasets/wj9rwkp9c2/1)](https://data.mendeley.com/datasets/wj9rwkp9c2/1/files/2eb60cac-96b8-46ea-b971-6415e972afc9)

### Al-Kindy Daibetes Data Description

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/92540bb0-8c76-4e27-bcd2-196c0d4141a2)


 ### Data Preperation: Vanderbilt Diabetes

-  Two(2) additional features(columns) has been added to the Vanderbilt datasets: CLASS and BMI
 
  1. CLASS - Using glycohemoglobin readings, Diabetic Screening Guidelines & Rules were used to classify patients as diabetic, pre-diabetic, or non-diabetic. Diabetic Screening Guidelines & Rules states that 
 
       - Non-diabetes  patients has  = below 5.7% glycohemoglobin readings
       - Pre-diabetes patients has  = 5.7 and 6.4% glycohemoglobin readings
       - Diabetes patients has = above 6.4% glycohemoglobin readings
   
  2.  BMI  was calculated using  =  703 x weight (lbs)/ [height(inches]2  
  
      - Weight- 2 sets of Weight data was provided in the original datasets : Weight1 and Weigh2,  The avergae weights was used in the calcaulation of BMI 

#### Vanderbilt Diabetes Data Description

   ![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f33de591-140d-42e0-8343-bc50f8b4cabf)

     
### Techniques:
- Machine Learning Algorithms that will be used are:  Decision Tree, Random Forest, K-Nearest Neighbors, Naive Bayes, Multinomial Logistic Regression and Support Vector Machines.
  

### Relevant tools:
- Python- for data analysis and visual representation of the datasets.


#### 1- Data Downloading and Inspection
   
Downloading datasets:



#### Libraries Used
1. Data-profiling
2. Sweetviz
3. Pandas
4. Imblearn
5. Numpy
6. Matplotlib
7. Seaborn
8. Sklearn
   

``` Python
!pip install ydata-profiling
!pip install imbalanced-learn
from ydata_profiling import ProfileReport
!pip install sweetviz
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import roc_curve, classification_report, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree # Decision Tree
from sklearn.metrics import make_scorer, f1_score,precision_score, recall_score, roc_auc_score, auc, roc_curve, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sklearn
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.svm import SVC # Support Vector Machines
from sklearn.neighbors import KNeighborsClassifier as knn

```

#### Storing the  Al-Kindy Diabetes dataset

``` Python
from google.colab import files
diabetes = files.upload()

data = pd.read_csv(r"Dataset_of_Diabetes_Al-Kindy.csv")
data_backup = data.copy()

```
``` Python
data.head()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/96799704-fa28-4ee0-a8bd-6d9e470a25a6)


#### Data Observations

``` Python
data.info()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/9e8c9f7a-4c22-45c7-8958-c751d744958c)

``` Python
data.shape
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a9741973-d3d9-41ea-8ab9-4fca06682161)

``` Python
d_class = data['CLASS'].value_counts()# show the counts of Non-diabetes Pre-diabetes and Diabetes
d_class
```


![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f144c544-de9b-4e35-8cb1-fb3e13d2d46d)

``` Python
data.describe()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ef95d59d-5038-472b-869e-5e2ee77447ec)



#### 2- Exploratory Data Analysis and Data Visualization - Al- Kindy Diabetes Datasets
1. Ydata Profiling
2. Sweetviz

``` Python
profile_data = ProfileReport(data)
profile_data
profile_data.to_file('Diabetes-Al_Kindy_Teaching_Hospital_data.html')

analyze_report = sv.analyze(data)
report_Al_Kindy_Diabetes = sv.analyze(data)
```
### Data Preprocessing

#### Checking for duplicate
``` Python
data.shape[0]
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/9ae06c9e-cf44-491a-8b4e-e9c649363367)
``` Python
data.duplicated().sum()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/146dfe84-04da-47cc-b015-8903a6d31c9a)
``` Python
data['ID'].nunique()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/d9ca7200-6c89-45fb-99ba-33dc9b6ac6e7)
``` Python
data.shape[0] - data['ID'].nunique()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3bab184a-6e50-4146-ad28-adffcc1892e2)
``` Python
data.duplicated(subset = ['ID']).sum()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ad85bc8e-e910-4060-8e44-9f8e385c11e7)

### Removing the Non-biological features
``` Python
df = data.drop(['ID'], axis=1)
df
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3f0ffa22-a3f0-4ffa-945d-4dcfdf867e2a)


#### Checking the Class count and Gender Count
``` Python
dd_class = df['CLASS'].value_counts()# show the counts of Non-diabetes Pre-diabetes and Diabetes
dd_class
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/cb79f340-0305-4d11-bed4-0d1d20557313)

``` Python
dd_Gender = df['Gender'].value_counts()# show the counts of Males and Females
dd_Gender
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a69b1928-01fe-483a-94e3-b2613efe5cb1)

#### Encoding the Class 'N': 1, 'P' : 2, 'Y' : 3

``` Python
class_encode = {'N': 1, 'P' : 2, 'Y' : 3}
df['CLASS'] = df['CLASS'].replace(class_encode)
```
### Changing the datatypes for CLASS and Gender
``` Python
df['CLASS'] = df.CLASS.astype('category')
df['Gender'] = df.Gender.astype('category')
df.info()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/34975a25-d425-49ee-85f3-a01b49024e36)


#### Median value of Non-diabetes, Pre-diabetes and Diabetes
``` Python
df.groupby('CLASS').median()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b18f8d83-3639-4cc4-801c-2f65d290bd43)
``` Python
df.describe()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/c7bf05dd-51d1-4e5c-b3b2-5a025806c438)

#### Seperating the data and labels
``` Python
X = df.drop(columns = 'CLASS', axis = 1)
Y = df['CLASS']
```
``` Python
print(X)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8d378d1e-7e61-4fb1-9127-3691b2a42765)
``` Python
print(Y)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/25cb1be2-4693-4f7f-8841-9681ec8f9244)

!pip install ydata-profiling
from ydata_profiling import ProfileReport
import pandas as pd
from google.colab import files
Dataset_of_Diabetes = files.upload()
data = pd.read_csv(r"Dataset_of_Diabetes_Al-Kindy.csv")
data
data.shape
data.info()
data.isnull().sum()
data.describe()
d_class = data['CLASS'].value_counts()# show the counts of Non-diabetes Pre-diabetes and Diabetes
d_class
profile_data = ProfileReport(data)
profile_data
profile_data.to_file('Diabetes-Al_Kindy_Teaching_Hospital_data.html')
analyze_report = sv.analyze(data)
report_Al_Kindy_Diabetes = sv.analyze(data)
```
Dataset 2 - Vanderbilt
1. Read and understanding  Vanderbilt_Diabetes_Dataset.csv 
2. View datasets
3. Vheck data shape 
4. Check datatypes 
5. Check for missing values 
6. Check data summary: Minimum, Maximum, Mean, and the Percentiles (25%, 50%, and 75%) of the datasets.
7. Run Ydata Profiling on the datasets.
8. Run EDA using Sweetviz for comparison with Ydata profiling

``` Python


from google.colab import files
Vanderbilt_Diabetes_ = files.upload()
data = pd.read_csv(r"Vanderbilt_Diabetes_Dataset.csv")
data
data.shape
data.info()
data.isnull().sum()
data.describe()
profile_data = ProfileReport(data)
profile_data
profile_data.to_file('Vanderbilt_Diabetes_Dataset.html')
analyze_report = sv.analyze(data)
report_Al_Kindy_Diabetes = sv.analyze(data)
```
### Result 
#### Dataset Al-Kindy Diabetes
- Number of observations 1000 with no Missing value, this consist of 565 Males and 435 Females between the age range of 20-79years. 
- Diabetes classes:   844 are diabetes, 103 are Non-diabates and 53 are prediabetes. Dataset is highly imbalance. 
- Normaly distributed variables, i.e. with skewness values between -0.5 and 0.5 : HbA1c, BMI.
- Slightly Skewed variables, i.e with Skewness values within the range of -1 and -0.5 (negative skewed) or 0.5 and 1(positive skewed): Age, Chol.
- Highly skewed vairables Skewed variables, i.e, skewness values less than -1 (negative skewed) or greater than 1 (positive skewed) : Urea, Cr, TG, HDL, LDL, VLDL.
- Highly Correlated:  CLASS with BMI and hBA1c; TG with VDL; and Urea with Cr.

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/4dc2c1cd-c091-41c8-94af-7e7d85e5c1c3)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3b37f2fb-764c-40a2-bca2-ba8b3e0a0b41)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/bf2cb794-f980-4548-9ad2-582056f249ab)


#### Vanderbilt Diabetes Datasets
- Number of observations 390 with 162 Males and 228 Females between the age range 19-92years.
- Diabetes classes: 67 are Diabetes, 298 are Non-diabetes, and 25 are Pre-diabetes.  Dataset is highly imbalance.
- Missing cells: Chol = 1, HDL = 1, Ratio =1, Height = 3, BMI = 3, Frame = 11, Waist = 2, Hip = 2, time.ppn = 3, bp.1s= 5, bp.1d= 5, bp.2s=252 and  bp.2d= 252

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/36b04502-30cf-42fc-a18b-9e0c486b7037)

Since there are 252 missing values  in bp.2s and bp.2d - we drop the variables, as bp.1s and bp.1d provided the systolic blood pressure and diastolic blood pressure needed for our analysis. 

``` Python
# To remove bp.2s and bp.2d from the dataframe
data3 = data2.drop(['bp.2s', 'bp.2d'], axis=1)
data3
profile_data3 = ProfileReport(data3)
profile_data3
```

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/6d697ee5-e69b-4153-8eb6-b04cb7c611f5)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/40701180-54b2-42ef-9f99-078faa9a2093)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/791bcd21-b519-4a88-b9e6-ee897fd0afe4)


### Data Processing
``` Python
## encoding the CLASS
class_encode = {'N': 1, 'P' : 2, 'Y' : 3}
data['CLASS'] = data['CLASS'].replace(class_encode)
# dropping Non biological varibles Patient ID and Number of Patients
data1 = data.drop(['ID', 'No_Pation'], axis=1)
data1
df = data1.drop_duplicates() # checking for duplicates
df.shape
df_class = df['CLASS'].value_counts()# show the counts of Non-diabetes(1)= 96, Pre-diabetes(96) and Diabetes(3) = 690
df_class
```

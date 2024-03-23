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
 
       - Non-diabetes  patients have  = below 5.7% glycohemoglobin readings
       - Pre-diabetes patients have  = 5.7 and 6.4% glycohemoglobin readings
       - Diabetes patients have = above 6.4% glycohemoglobin readings
   
  2.  BMI  was calculated using  =  703 x weight (lbs)/ [height(inches]2  
  
      - Weight- 2 sets of Weight data was provided in the original datasets : Weight1 and Weigh2,  The avergae weights was used in the calcaulation of BMI 

#### Vanderbilt Diabetes Data Description

   ![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f33de591-140d-42e0-8343-bc50f8b4cabf)

     
### Techniques:
- Machine Learning Algorithms that will be used are:  Decision Tree and Random Forest.
  

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
from sklearn.feature_selection import VarianceThreshold,  SelectFromModel
from sklearn.preprocessing import  MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree # Decision Treefrom sklearn.metrics import make_scorer, f1_score,precision_score, recall_score, roc_auc_score, auc, roc_curve, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

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


### Data Observations
From the information shown below, the data has 1000 rows, 14 columns and has no null values which suggests that this database does not have missing values. 

Additionally the info also tells us that the column ID, No of Patients and Cr are of type int, the column Gender and CLASS are of  type object, the column Urea, HbA2c, Chol, TG, HDL,LDL, VLDL,BMI are type float. 

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
Non-biological variables in the dataset were dropped ('id', 'Location', 'frame', 'time.ppn) becuase it will not provide any information on patients classes.

``` Python
df = data.drop(['ID', 'No_Patients'], axis=1)
df
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/5c700a9e-30c9-4844-abf9-0d99e9d6ab5f)


#### Checking the Class count and Gender Count
This provide insights on the numbers of Males and Females in the diabetes classes
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
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b85127a0-c580-446a-b30e-8e181385f2d1)



#### Median value of Non-diabetes, Pre-diabetes and Diabetes
``` Python
df.groupby('CLASS').median()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/411cdf37-4b58-456b-bee9-691fed1fe092)

It was observed that the dataset has outliers

``` Python
df.describe()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/5a9cf075-c0d1-4cae-9e3d-87b3a0632575)


## Seperating the data and labels
The main objective is to determine the some of the possible factors that cause diabetes that are made available to us from this dataset and to create machine learning models that will do this for us as well. In order to get best visual results and to understand certain visual and statistical trends we would need to seperate the data into dependent variables and independent variables
- 1 -The independent variables will be the features of diabetes patients : Gender, AGE, Urea, Cr, HbA1c, TG, Chol, HDL, VLDL, BMI.
- 2- The independent variables will be the CLASS of diabetes patients.

``` Python
X = df.drop(columns = 'CLASS', axis = 1)
Y = df['CLASS']
```
``` Python
print(X)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/e1cb1586-c3d0-49e4-91b6-dfc709202791)

#### Ydata profiling 2 was performed 

``` Python
profile_df = ProfileReport(df)
profile_df
``` 
#### Onehot ecoding Gender
One hot encoding:  was performed on the feature ‘Gender’ because it is a categorical variable containing label (Males and Females) values rather than numeric values. One hot encoding is performed because machine learning algorithms cannot operate on label data directly. They require all input variables and output variables to be numeric.

``` Python
X = pd.get_dummies(X, columns = ['Gender'], prefix = ['Gender'])
X.head()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3c9da75d-806b-4398-98c7-0d2089750c37)

``` Python
print(Y)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/25cb1be2-4693-4f7f-8841-9681ec8f9244)

#### Pie Chart of CLASS features
``` Python
df['CLASS'].value_counts()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ff32a0c9-b6c2-4132-88e6-e12f3048b39c)

``` Python
fig, axes =plt.subplots(nrows =1, ncols=2, figsize =(10, 4))

pie_colors = ['skyblue', 'lightcoral', 'Green']
axes[0].pie(df['CLASS'].value_counts(), labels =df['CLASS'].value_counts().index, autopct='%1.1f%%', startangle=90, colors=pie_colors)
axes[0].set_title('Class Distribution Before Resampling(Pie Chart)')

countplot_colors = sns.color_palette(pie_colors)
sns.countplot(x='CLASS', data=df, palette=countplot_colors, ax=axes[1])
axes[1].set_title('Classfication Distribution Before Resampling(count plot)')

plt.tight_layout()
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/5f24b6b9-c739-44fc-b387-ce839323381e)

#### Using Dummies to represent Gender in the datasets
``` Python
X = pd.get_dummies(X, columns = ['Gender'], prefix = ['Gender'])
X.head()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/39963e1c-9669-4275-8d13-049f308fd058)

# Models Evaluation
To evaluate models, data frame will be split into training (XX_train, YY_train) and testing (XX_test, YY_test) sets.
Before the application of the Machine Learning Algorithm, the listed observed issues will be treated:
1-	Balancing of data frame
2-	Outlier treatment
3-	Feature scaling of the data frame

####  Splitting datasets
Splitting the data into two parts, the first part contains 80% of the data whereas the second part of the data contains the remaining 20% of the data. We do this to avoid over-fitting, the two parts are called training and test splits, which gives us a better idea as to how our algorithm performed during the testing phase.The training split gets 80% of the data and the test split has 20% of the data.

``` Python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, stratify= Y, random_state= 11)
```
``` Python
def plot_resampling_results(Y_resampled, title):
  plt.figure(figsize = (10, 4))
  pd.Series(Y_resampled).value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'Green'])
  plt.title(title)
  plt.show()
``` 

#### Balancing dataset
Balancing the dataset: it was observed that the dataset was imbalanced from the exploratory data analysis carried out. Several balancing techniques were carried out.
1-	Random Under sampling Technique.
2-	SMOTE technique.
3-	Combination of SMOTE and Tomek Link Technique.
From the above technique, the Combination of SMOTE and Tomek Link Technique was selected as it combines the SMOTE ability to generate synthetic data for the minority class and Tomek Link's ability to remove the data that are identified as Tomek links from the majority class (that is, samples of data from the majority class that is closest with the minority class data).

#### Random Undersampling
Random undersampling involves randomly selecting examples from the majority class to delete from the training dataset.
This has the effect of reducing the number of examples in the majority class in the transformed version of the training dataset. This process can be repeated until the desired class distribution is achieved, such as an equal number of examples for each class.
This approach may be more suitable for those datasets where there is a class imbalance although a sufficient number of examples in the minority class, such a useful model can be fit.

``` Python
rus = RandomUnderSampler(random_state =101)
X_rus, Y_rus = rus.fit_resample(X_train, Y_train)
plot_resampling_results(Y_rus, 'Class Distribution After Random Undersampling')
``` 

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/97b8ccba-20dc-4a27-996f-f2480d6315b9)
``` Python
Y_train.value_counts()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/d07a44b9-d0cf-417b-9ea2-b8269d6a53dc)
``` Python
Y_rus.value_counts()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/e7948829-abdf-4e10-a072-812d101e130d)
``` Python
print('No. of records removed:', Y_train.shape[0] - Y_rus.shape[0])
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a2121dac-c4d8-4a11-819d-150b35153812)

### SMOTE Technique
Smote generates synthetic minority class examples by interpolating between existing instances. This helps in increasing the diversity of the minority class.
SMOTE is an oversampling technique where the synthetic samples are generated for the minority class. This algorithm helps to overcome the overfitting problem posed by random oversampling. It focuses on the feature space to generate new instances with the help of interpolation between the positive instances that lie together.

``` Python
smote = SMOTE(random_state =123)
X_smote, Y_smote = smote.fit_resample(X_train, Y_train)
plot_resampling_results(Y_smote, 'Class Distribution After SMOTE')
``` 

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3e34f268-43f5-460f-98e0-d6dbe2329b45)
``` Python
Y_train.value_counts()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/69596304-7176-497e-b931-18a591f51332)

``` Python
Y_smote.value_counts()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/506b32b6-00b4-4681-8823-c568b1306524)

``` Python
print('No. of records added:', Y_smote.shape[0] - Y_train.shape[0])
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8525e62e-3ebd-4199-8548-fbaf6da52650)

#### Combination of SMOTE and Tomek Links Technique
The process of SMOTE-Tomek Links is as follows. Start of SMOTE: choose random data from the minority class. Calculate the distance between the random data and its k nearest neighbors. Multiply the difference with a random number between 0 and 1, then add the result to the minority class as a synthetic sample. SMOTE-Tomek uses a combination of both SMOTE and the undersampling Tomek link. Tomek link is a cleaning data way to remove the majority class that was overlapping with the minority class. 
SMOTETomek which combines both oversampling (using SMOTE for the minority class) and undersampling (using Tomek links to remove Tomek pairs). This combined approach aims to create a more balanced dataset. Tomek link is a cleaning data way to remove the majority class that was overlapping with the minority

``` Python
smote_tomek = SMOTETomek(random_state = 20)
X_st, Y_st = smote_tomek.fit_resample(X_train, Y_train)
plot_resampling_results(Y_st, 'Classification After SMOTE and Tomek Links')
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/fd300379-b5d8-4a86-824f-121460225733)
``` Python
Y_train.value_counts()
``` 
``` Python
Y_st.value_counts()
``` 
``` Python
print('No. of records added:', Y_st.shape[0] - Y_train.shape[0])
``` 

#### Plotting Boxplot to visualize the outliers present in the dataframe
When analyzing data, identifying and addressing outliers is crucial. These anomalies can skew results, leading to inaccurate insights and decisions.

``` Python
plt.figure(figsize = (10, 10))
X_st[['AGE', 'Urea', 'Cr','HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']].boxplot(vert =0)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f9f12bf3-437f-485b-b7ec-f26d4f09f8b8)



#### Treating the Outliers
``` Python
def  replace_outlier(col):
  Q1, Q3 =np.quantile(col, [.25, .75])
  IQR =Q3 -Q1
  LL =Q1 -1.5*IQR
  UL = Q3 + 1.5*IQR
  return LL, UL # Winsorization -UL - Capping, LL - Flooring
  
df_num = X_st[['AGE', 'Urea', 'Cr','HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']]

for i in df_num.columns:
  LL, UL = replace_outlier(df_num[i])
  df_num[i] = np.where(df_num[i]> UL, UL, df_num[i])
  df_num[i] = np.where(df_num[i] < LL, LL, df_num[i])  # Winsorization - Capping and Flooring
``` 

#### Plotting Boxplot to visualize the dataframe after treating the Outliers
``` Python
plt.figure(figsize = (15, 10))
df_num.boxplot(vert=0)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8e4315b7-e088-4193-9ca6-175f5c173be4)


#### Plotting the heatmap to see the relatonship between the features
``` Python
plt.figure(figsize= (10,10))
sns.set(font_scale = 1.0)
sns.heatmap(df_num.corr(), annot =True)
```

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3d630016-3412-415d-876c-7405555c0e7e)


### Min Max Scaling
Since  dataset contains features that have different ranges, units of measurement, or orders of magnitude.  These variation in feature values can lead to biased model performance or difficulties during the learning process. MinMax Scaling technique was employed.   this compresses all the ouliers in the narrow range between [0,1].This process enhances data analysis and modeling accuracy by mitigating the influence of varying scales on machine learning models.

 
``` Python
scaler = MinMaxScaler().fit(df_num)
print(scaler)
scaler.transform(df_num)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ec45c202-35ad-4247-aaf7-f9e93acfadc6)
``` Python
X_st_scaled = scaler.transform(df_num)
print(X_st_scaled)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b451034a-f5d9-4119-86ac-2d509bceb6cc)

## Random Forest Classifier

Random Forest: can be defined as a collection of tree-type classifiers. It uses simple probability to select the strongest features for its inputs. The advantages of RF include:
1-	handling missing values (missing data) in the dataset.
2-	producing lower error i.e. improves performance by reducing variance.
3-	effectively handling large amounts of training data efficiently i.e. resistant to irrelevant features.
4-	providing good classification results and avoiding overfitting

``` Python
fr =RandomForestClassifier(n_estimators=500, random_state =11)
fr.fit(X_st_scaled, Y_st)
importances = fr.feature_importances_
print(importances)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/1badcd17-af80-46dd-92c5-ca83db8009f3)

``` Python
indices = np.argsort(importances)[::-1]

plt.ylabel('Feature importance')
plt.bar(range(X_st_scaled.shape[1]),
importances[indices],
align ='center')

feat_labels = X.columns
plt.xticks(range(X_st_scaled.shape[1]),
           feat_labels[indices], rotation = 90)
plt.xlim([-1,X_st_scaled.shape[1]])
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/49022492-4b75-4501-a7e2-66f6b37502a1)

#### Prediction
In machine learning, to build a predictive model for either classification or regression tasks, a dataset is split into two different parts: training and testing. 
he training part is used to train the machine learning model whereas the testing part is used for predictions by the model. These predictions are then evaluated using different evaluation methods and in this project we will be using Random forest and Decision Trees.

``` Python
Y_pred_fr = fr.predict(X_test)
Y_pred_fr
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/63a8358b-a649-402c-8a3f-4abb8d84f7b5)

We'll be using "classification_report" to measure the quality of our predictions for each algorithm i.e. how many predictions are True and how many are False.
Now let's define some of the terms in "classification_report" that will be used to analyze each model
Accuracy: Fraction (or percentage) of predictions that were correct.
Precision: Fraction (or percentage) of correct predictions among all examples predicted to be positive, meaning what percent of our predictions were correct.
Recall: Fraction (or percentage) of correct predictions among all real positive examples. In simple terms, what percent of the positive cases did we catch properly.
F1-Score: Weighted harmonic mean of precision and recall. In simple terms, what percent of positive predictions were correct.

### Accuracy
``` Python
print(metrics.accuracy_score(Y_test, Y_pred_fr ))
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a9fc3684-9eae-4ea9-802d-30d9acd60366)


``` Python
f1 = f1_score(Y_test,Y_pred_fr, average= 'weighted')
print(f'F1 Score:{f1:.2f}')
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3564dacb-903d-4364-b5ba-fc5558cbc40f)

#### Confusion Matrix
``` Python
cm = confusion_matrix(Y_test,Y_pred_fr)
print("Confusion Matrix:")
print(cm)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/1681f7a3-ac8e-4c33-b152-01b0f77db35d)

``` Python
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '2', '3'])
cm_disp.plot()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8253257f-0e3d-48f7-8f01-ae353e17b108)

#### ROC CURVE 
``` Python
Y_pred_fr_proba = fr.predict_proba(X_test)
Y_pred_fr_proba.shape
Y_pred_fr .shape
roc_auc_score(Y_test,Y_pred_fr_proba, multi_class='ovr')
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/07bd77f2-c839-4eb9-84e6-ab349f837e75)



## Decision Tree on Al-Kindy Diabetes Datasets

Decision Tree: is a non-parametric supervised learning algorithm for classification and regression tasks. That can handle numerical & categorical data and multi-output problems. Its

1-	validate a model using statistical tests. That makes it possible to account for the reliability of the model.
2-	performs well even if its assumptions are somewhat violated by the true model from which the data were generated.


``` Python
clf_ddt = DecisionTreeClassifier(random_state =42)
clf_ddt.fit(X_st_scaled, Y_st)
importances_ddt = clf_ddt.feature_importances_
print(importances_ddt)
```

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/10a452f6-90c5-4887-ac7a-85a60ab382a5)

``` Python
indices_ddt = np.argsort(importances_ddt)[::-1]

plt.ylabel('Feature importance')
plt.bar(range(X_st_scaled.shape[1]),
importances_ddt[indices_ddt],
align ='center')

feat_labels = X.columns
plt.xticks(range(X_st_scaled.shape[1]),
           feat_labels[indices_ddt], rotation = 90)
plt.xlim([-1,X_st_scaled.shape[1]])
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f7dae258-551e-42d6-b90a-66c3d0b328fe)

``` Python
Y_pred_ddt = clf_ddt.predict(X_test)
Y_pred_ddt
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/edcdac83-675c-49bd-bc91-eeca2eaeee3b)

#### f1 score
``` Python
f1_ddt = f1_score(Y_test,Y_pred_ddt, average= 'weighted')
print(f'F1 Score:{f1:.2f}')
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ac90fc43-ad28-4c1f-a400-7ccb10120585)


#### ROC CURVE
``` Python
Y_pred_ddt_proba = clf_ddt.predict_proba(X_test)
Y_pred_ddt_proba.shape
Y_pred_ddt.shape
roc_auc_score(Y_test,Y_pred_ddt_proba, multi_class='ovr')
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3aaf868c-980b-41dc-ab98-9bb82f1b1ebb)

#### Confusion matrix
``` Python
cm_ddt = confusion_matrix(Y_test,Y_pred_ddt)
print("Confusion Matrix:")
print(cm_ddt)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/86cde74f-2ed4-415d-b900-ced88893db0e)

``` Python
cm_disp_tr = ConfusionMatrixDisplay(confusion_matrix=cm_ddt, display_labels=['1', '2', '3'])
cm_disp_tr.plot()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a84b61e6-6344-4e45-a3f0-2fbba77b26ed)

## Uploading dataset 2- Vanderbilt Diabetes Dataset
``` Python
from google.colab import files
Vanderbilt_Diabetes = files.upload()

data2 = pd.read_csv(r"Vanderbilt_Diabetes_Dataset.csv")
data2_backup = data2.copy()
```
### Viewing dataset
``` Python
data2
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/30d95773-5c34-4d04-a73d-5db3a7df7686)

## Data  Observations
``` Python
data2.info()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0b304325-a1be-489f-9684-82fed00a9c75)
``` Python
data2.shape
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0c4a3ab6-f45f-4ffa-b4d4-ea157a88d8ca)
``` Python
data2.isnull().sum()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/2f4af577-b75f-4d8e-ac5b-b950ca98ff54)

``` Python
dd_class = data2['CLASS'].value_counts()
dd_class
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0205dc1d-9f89-496d-9bc9-7bbae360e558)
``` Python
data2.describe()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/547700c6-9e8f-400f-a104-5b1767f967ee)


## Checking for duplicate
``` Python
data2.shape[0]
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/9f3f49c6-9158-434b-8f31-d96019b75a9d)
``` Python
data2.duplicated().sum()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/079309ab-b62b-4ab5-9582-31e6df0fd79a)
``` Python
 data2['id'].nunique()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f5727975-f644-4167-ae78-cdfbd28500e1)


 ### Removing the Non-biological features
 Non-biological variables in the dataset were dropped ('id', 'Location', 'frame', 'time.ppn) becuase it will not provide any information on patients classes

 
 ``` Python
data3 = data2.drop(['id', 'Location', 'frame', 'time.ppn'], axis=1)
data3
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/69ec5f38-6931-4c45-a113-6789adbe69c0)

### Removing the bp.2s and bp.2d as over 50% of the cells are missing.

Dropping  bp.2s and bp.2d as 65% of the cells were missing.  
1-	The threshold used: drop variables if 50% of cells are missing.
2-	Note: The dataset has bp.1s and bp.1d, which provides information on the Systolic and Diastolic blood pressure of the patients

 ``` Python
dff = data3.drop(['bp.2s', 'bp.2d'], axis=1)
dff
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3cd8aeb5-7aa5-4a3b-a7da-958860734a7f)

### Renaming bp.1s: Systolic_Blood_Pressure and bp.1d:Diastolic_Blood_Pressure
For easy reading of the features, bp.1s and bp.1d was renamed to Systolic blood pressure and Diastolic blood pressure.

 ``` Python
df1 = dff.rename(columns = {'bp.1s': 'Systolic_Blood_Pressure','bp.1d':'Diastolic_Blood_Pressure' })

```

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/c82d727e-8496-4018-9fa5-2bf2d0d6aa4a)

#### Checking the Class count and Gender Count
 ``` Python
d_class = df1['CLASS'].value_counts()
d_class
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/21ee327a-999d-40f9-a6f4-a6ff135ac4c1)

 ``` Python
d_Gender = df1['Gender'].value_counts()
d_Gender
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/7fc65a3f-9c93-4f0e-a905-7687ac8c991b)

### Encoding the Class 'N': 1, 'P' : 2, 'Y' : 3
Models only work with numerical values. For this reason, it is necessary to convert the categorical values of the features into numerical ones, So the machine can learn from those data,   extract valuable information and gives the right model. 

To ensure that Class variables and Gender were read categorical variables, the datatype was changed

 ``` Python
clas_encode = {'N': 1, 'P' : 2, 'Y' : 3}
df1['CLASS'] = df1['CLASS'].replace(clas_encode)
df1['CLASS'] = df1.CLASS.astype('category')
df1['Gender'] = df1.Gender.astype('category')
df1.info()
 ```

### Median value of Non-diabetes, Pre-diabetes and Diabetes

1 = Non-diabetes; 2 = Pre-diabetes; 3 = Diabetes

The median class for all the variables was checked. This provides information on the diabetes classes
It was observed that:
1- 	Prediabetes patients have a cholesterol = 204, stab.glu = 98, glyhb = 6.13, Age = 61 and Systolic_Blood_Pressure = 140
2-	Diabetes patients have cholesterol = 218, stab.glu = 41,  glyhb =9.37 , Age = 59 and Systolic blood pressure = 146

df1.groupby('CLASS').median()

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/e93dda0f-9b99-4679-b138-34aed415578a)

### Ydata profile 2
Another Ydata profiling was performed as the shape of the dataframe is now different because some features were removed,
 ``` Python
profile_df1 = ProfileReport(df1)
profile_df1
profile_df1.to_file('Vanderbilt_Diabetes_data2.html')
 ```
### Sweetviz Profile
 ``` Python
analyze_reportt = sv.analyze(df1)
report_Vanderbilt_Data2 = sv.analyze(df1)
 ```
## Seperating the data and labels

``` Python
XX = df1.drop(columns = 'CLASS', axis = 1)
YY = df1['CLASS']
print(XX)
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/97b2f53e-c59b-42e9-b310-7db4c6dffce8)


# Model Evaluation
To evaluate models, dataset was split into training (XX_train, YY_train) and testing (XX_test, YY_test) sets.
Before the application of the Machine Learning Algorithm, the listed observed issues will be treated
1-	Onehot encoding of variable (Gender)
2-	Handling missing cells 
3-	Balancing of dataset 
4-	Outliers’ treatment
5-	feature scaling of dataset 

### Onehot - creating seperate column for female and male

``` Python
XX = pd.get_dummies(XX, columns = ['Gender'], prefix = ['Gender'])
XX.head()
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/db1fc9fb-9674-4262-88db-35569ede2435)
``` Python
print(YY)
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/46626e93-9587-4f1f-9679-b0d7d82c6c82)

### Pie Cahart of the CLASS

``` Python
fig, axes =plt.subplots(nrows =1, ncols=2, figsize =(10, 4))

pie_colors = ['Blue', 'Red', 'Green']
axes[0].pie(df1['CLASS'].value_counts(), labels =df1['CLASS'].value_counts().index, autopct='%1.1f%%', startangle=90, colors=pie_colors)
axes[0].set_title('Class Distribution Before Resampling(Pie Chart)')


countplot_colors = sns.color_palette(pie_colors)
sns.countplot(x='CLASS', data=df1, palette=countplot_colors, ax=axes[1])
axes[1].set_title('Classfication Distribution Before Resampling(count plot)')

plt.tight_layout()
plt.show()
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/02a92c92-e736-4474-9adb-43693168f607)


### Splitting datasets
``` Python
XX_train, XX_test, YY_train, YY_test = train_test_split(XX, YY, test_size=0.20, stratify= YY, random_state= 11)
 ```
## Handling missing cells

Following the split of the data into training and test sets, it was discovered that some cells were missing. To address this issue, a condition was applied to the split which resulted in the missing cells being moved from the training set to the testing set. This condition was necessary to ensure that the missing cells were handled correctly during the training process, and to prevent any errors from occurring during prediction.

#### Checking missing cells in training set and test set
``` Python
XX_train.isnull().sum()
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/99a27315-6757-430c-a91d-21a01506b258)
``` Python
XX_test.isnull().sum()
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/82ba4b52-f702-4629-a031-b7c2a5efe3d6)

##### Identify missing cell in testing set
``` Python
missing_cells_mask =  XX_test.isnull().any(axis =1)
 ```
##### Moving missing cells to training sets
``` Python
XX_train_missing = XX_test[missing_cells_mask]
YY_train_missing = YY_test[XX_train_missing.index]
``` 
##### Removing rows with missing values from the testing set
``` Python
XX_test = XX_test.dropna()
YY_test = YY_test.loc[XX_test.index]
 ```
##### Concatenate the training set with the row containing missing values
``` Python
XX_train = pd.concat([XX_train, XX_train_missing])
YY_train = pd.concat([YY_train, YY_train_missing])
```
##### Confirming all missing cells are in training sets
``` Python
XX_train.isnull().sum()
XX_test.isnull().sum()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/68c54125-9b02-4aaa-8507-3561e0b55c8f)

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/03ff071c-e56a-415e-9c5b-e279e8a44a25)

#### Confirming test set does not have missing cells
``` Python
XX_test.isnull().sum()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/63b92c82-8d5e-4b77-b642-5eba66c8946c)


### Using median and mean to replace the missing cells in the data
The exploratory analysis helped in determining the appropriate imputation method by observing the feature distribution

From the EDA above:
chol = skewness =0.95- slightly skewed - median will be used to replace missing cells
hdl = skewness =1.23- Highly skewed -median will be used to replace missing cells
ratio = skewness =2.24- Highly skewed- median will be used to replace missing cells
Height = skewness = 0.016: normally distributed- mean will be used to replace missing cells
BMI = skewness =0.84- slightly skewed- median will be used to replace missing cells
Systolic_Blood_Pressure = 1.09- Highly skewed -median will be used to replace missing cells
Diastolic_Blood_Pressure = 0.25: normally distributed- mean will be used to replace missing cells
waist = 0.46: normally distributed- mean will be used to replace missing cells
hip =0.79- slightly skewed -median will be used to replace missing cells

``` Python
XX_train =  XX_train.fillna({'chol' : XX_train['chol'].median(),
                             'hdl' : XX_train['hdl'].median(),
                             'ratio' :XX_train['ratio'].median(),
                             'Height' : XX_train['Height'].mean(),
                             'BMI' :XX_train['BMI'].median(),
                             'Systolic_Blood_Pressure' : XX_train['Systolic_Blood_Pressure'].median(),
                             'Diastolic_Blood_Pressure' :XX_train['Diastolic_Blood_Pressure'].mean(),
                             'waist' : XX_train['waist'].mean(),
                             'hip':XX_train['hip'].median() },
                            inplace = True)

```
``` Python
XX_train.isnull().sum()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/551804a2-eb49-4643-8931-b11768365a1d)


## Balancing the training sets

Balancing the dataset: it was observed that the dataset was imbalanced from the exploratory data analysis carried out. Several balancing techniques were carried out.
1-	Random Under sampling Technique.
2-	SMOTE technique.
3-	Combination of SMOTE and Tomek Link Technique.
From the above technique, the Combination of SMOTE and Tomek Link Technique was selected as it combines the SMOTE ability to generate synthetic data for the minority class and Tomek Link's ability to remove the data that are identified as Tomek links from the majority class (that is, samples of data from the majority class that is closest with the minority class data).


``` Python
def plot_resampling_results(YY_resampled, title):
  plt.figure(figsize = (10, 4))
  pd.Series(YY_resampled).value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'Green'])
  plt.title(title)
  plt.show()
``` 
#### Technique 1:  Random Undersampling Vanderbilt Datasets
``` Python
russ = RandomUnderSampler(random_state =101)
XX_russ, YY_russ = russ.fit_resample(XX_train, YY_train)
plot_resampling_results(YY_russ, 'Class Distribution After Random Undersampling')
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/e2f9dc80-c299-4f11-8412-be62d5fb6422)

``` Python
YY_train.value_counts()

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/48a215ef-1a36-4bf7-b18c-9d2376f236bc)

``` Python
YY_russ.value_counts()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/36a4d1e3-b3a9-4a42-8f32-cde1dfaa0186)


``` Python
print('No. of records removed:', YY_train.shape[0] - YY_russ.shape[0])
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/08a51f15-b5a9-405b-a121-85795db50122)


#### Technique 2: SMOTE( Synthetic Minority Over-Sampling Technique)- Vanderbilt Datasets
Smote generates synthetic minority class examples by interpolating between existing instances. This helps in increasing the diversity of the minority class.
SMOTE is an oversampling technique where the synthetic samples are generated for the minority class. This algorithm helps to overcome the overfitting problem posed by random oversampling. It focuses on the feature space to generate new instances with the help of interpolation between the positive instances that lie together.

``` Python
smote = SMOTE(random_state =11)

XX_smote, YY_smote = smote.fit_resample(XX_train, YY_train)
plot_resampling_results(YY_smote, 'Class Distribution After SMOTE')
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ad26d3d8-9185-4ce2-8baf-3dfbe6563443)

``` Python
YY_train.value_counts()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/484d49c9-3b09-4621-a9aa-877cf58bcf5a)

``` Python
YY_smote.value_counts()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b8efdbb6-837b-4d16-a26f-c6f011583f70)

``` Python
print('No. of records added:', YY_smote.shape[0] - YY_train.shape[0])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/de0dc36e-1376-4acf-bc2d-d434c268cb71)

##### Technique 3: Combination of SMOTE and Tomek Links
The process of SMOTE-Tomek Links is as follows. Start of SMOTE: choose random data from the minority class. Calculate the distance between the random data and its k nearest neighbors. Multiply the difference with a random number between 0 and 1, then add the result to the minority class as a synthetic sample. SMOTE-Tomek uses a combination of both SMOTE and the undersampling Tomek link. Tomek link is a cleaning data way to remove the majority class that was overlapping with the minority class. 
SMOTETomek which combines both oversampling (using SMOTE for the minority class) and undersampling (using Tomek links to remove Tomek pairs). This combined approach aims to create a more balanced dataset. Tomek link is a cleaning data way to remove the majority class that was overlapping with the minority

``` Python
smote_tomek = SMOTETomek(random_state = 20)
XX_st, YY_st = smote_tomek.fit_resample(XX_train, YY_train)
plot_resampling_results(YY_st, 'Classification After SMOTE and Tomek Links')

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b7d5161c-856b-4df6-8aff-332ccad75753)

``` Python
YY_train.value_counts()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/e21a1f10-3e1a-4077-bb88-7b943924a134)


``` Python
YY_st.value_counts()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/cb237667-ef22-4111-a86d-639e12a733c1)

``` Python
print('No. of records added:', YY_st.shape[0] - YY_train.shape[0])
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/29658632-51f5-4047-a0df-3c0db412853c)


#### Plotting Boxplot to visualize the outliers present in the dataframe

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/79dc8abe-2d8c-42b0-82e9-599b0e8fa465)

#### Treating the Outliers
``` Python
def  replace_outlier(col):
  Q1, Q3 =np.quantile(col, [.25, .75])
  IQR =Q3 -Q1
  LL =Q1 -1.5*IQR
  UL = Q3 + 1.5*IQR
  return LL, UL # Winsorization -UL - Capping, LL - Flooring
```

``` Python
df_num = XX_st[[ 'chol', 'stab.glu', 'hdl','ratio', 'glyhb', 'Age', 'Height', 'Weight', 'BMI', 'Systolic_Blood_Pressure', 'Diastolic_Blood_Pressure','waist', 'hip', 'Gender_female', 'Gender_male']]

for i in df_num.columns:
  LL, UL = replace_outlier(df_num[i])
  df_num[i] = np.where(df_num[i]> UL, UL, df_num[i])
  df_num[i] = np.where(df_num[i] < LL, LL, df_num[i])  # Winsorization - Capping and Flooring
```
##### Plotting Boxplot to visualize the dataframe after treating the Outliers

``` Python
plt.figure(figsize = (15, 10))
df_num.boxplot(vert=0)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/cd617cc3-7e09-4923-bb2e-c32f97c3b874)

#### Plotting the heatmap to see the relatonship between the features
``` Python
plt.figure(figsize= (7,7))
sns.set(font_scale = 0.7)
sns.heatmap(df_num.corr(), annot =True)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/33484669-66c9-4309-9bc0-38bf735c8a5b)

waist and hip are highly positively correlated
BMI and Weight are  highly positively correlated
ratio and hdl are highly negatively correlated

there is a relationship between 
##### Features Scaling using MinMax Scaler

``` Python
scalerr = MinMaxScaler().fit(df_num)
print(scalerr)
scalerr.transform(df_num)
``` 

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b764cd56-6f72-4743-8122-80aa7c229edd)

``` Python
XX_st_scaled = scalerr.transform(df_num)
print(XX_st_scaled)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/68d742ad-b199-495f-b2a6-fd05e30c51ba)

## Decision Tree on Vanderbilt Diabetes Datasets

### Feature selection using the embedded technique- Random Forest

Random Forest: can be defined as a collection of tree-type classifiers. It uses simple probability to select the strongest features for its inputs. The advantages of RF include:
1-	handling missing values (missing data) in the dataset.
2-	producing lower error i.e. improves performance by reducing variance.
3-	effectively handling large amounts of training data efficiently i.e. resistant to irrelevant features.
4-	providing good classification results and avoiding overfitting

``` Python
forest = RandomForestClassifier( n_estimators=500, random_state =11)
forest.fit(XX_st_scaled, YY_st)
importances = forest.feature_importances_
print(importances)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/6392e7a2-4dbc-429d-9916-cebeb4611ed2)


``` Python
indices = np.argsort(importances)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_st_scaled.shape[1]),
importances[indices],
align ='center')

feat_labels = XX.columns
plt.xticks(range(XX_st_scaled.shape[1]),
           feat_labels[indices], rotation = 90)
plt.xlim([-1,XX_st_scaled.shape[1]])

``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ac86b832-08a7-4433-b243-7576c44d1fc2)

The first five important features are Gylhb, stab.glu, Age, chol,and Systolic Blood pressure

#### Prediction 
``` Python
YY_pred_forest = forest.predict(XX_test)
YY_pred_forest 
```

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/32087755-0e89-49c5-8ed8-da0734f3532d)
 ### checking accuracy
 ``` Python
 print(metrics.accuracy_score(YY_test, YY_pred_forest ))
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/dd4095f1-0946-47c3-9488-84afe791d2a7)
 Accuracy score is very low
 #### f1 score
``` Python
f1 = f1_score(YY_test,YY_pred_forest, average= 'weighted')
print(f'F1 Score:{f1:.2f}')
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/03b36c41-7711-410d-9907-aad96af98a42)
f1 score is extremely low
#### Confusion matrix
``` Python
cm = confusion_matrix(YY_test,YY_pred_forest)
print("Confusion Matrix:")
print(cm)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/1c069ceb-ce2a-464b-8844-cdf08fe6a699)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/d1c1ff8c-d701-4c08-bf06-eee8c4c4dfc6)

### ROC Curve
``` Python
YY_pred_proba = forest.predict_proba(XX_test)
YY_pred_proba.shape
``` 

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/4f6d58be-84f0-496c-a160-ded13de01b78)
``` Python
YY_pred.shape
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8c6f3c45-1cfc-491f-a94b-7302c58ba668)
``` Python
roc_auc_score(YY_test,YY_pred_proba, multi_class='ovr')

``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b219d69f-d49d-4498-bce5-70db56062b86)

## Decision Tree on Vanderbilt Diabetes Datasets

### Feature selection using the embedded technique- Decision Tree Classifier 
Decision Tree: is a non-parametric supervised learning algorithm for classification and regression tasks. That can handle numerical & categorical data and multi-output problems. Its

1-	validate a model using statistical tests. That makes it possible to account for the reliability of the model.
2-	performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

``` Python
clf_dt = DecisionTreeClassifier(random_state =42)
clf_dt.fit(XX_st_scaled, YY_st)
importances_dt = clf_dt.feature_importances_
print(importances)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/e1b3673a-8e16-4ffd-bcd8-9a63e262bd34)

``` Python
indices_dt = np.argsort(importances_dt)[::-1]

plt.ylabel('Feature importance')
plt.bar(range(XX_st_scaled.shape[1]),
importances_dt[indices_dt],
align ='center')

feat_labels = XX.columns
plt.xticks(range(XX_st_scaled.shape[1]),
           feat_labels[indices_dt], rotation = 90)
plt.xlim([-1,XX_st_scaled.shape[1]])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/99dd59d8-d01e-42e2-8278-9c72534c5c0d)

the importance features are : glyhb and chol
### Prediction
``` Python
YY_pred_dt = clf_dt.predict(XX_test)
YY_pred_dt
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/59343f12-4f4f-4c04-8556-bc8e545ea17c)

### Accuracy
``` Python
print(metrics.accuracy_score(YY_test, YY_pred_dt ))
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a6e18687-403f-4e07-8388-7faf0198400b)

Accuracy is same as random forest, extemeely low
``` Python
f1_dt = f1_score(YY_test,YY_pred_dt, average= 'weighted')
print(f'F1 Score:{f1:.2f}')

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/75a6e70d-7b3b-4905-9a3b-3897d2b2638a)

f1 score is also low
#### ROC Curve
``` Python
YY_pred_dt_proba = clf_tr.predict_proba(XX_test)
YY_pred_dt_proba.shape
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/02e8b94f-50ca-4ad3-a51a-80d3808c238d)
``` Python
YY_pred_dt.shape
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/cb7c750a-85bd-4b0d-bfd2-823ea61a1829)
``` Python
roc_auc_score(YY_test,YY_pred_dt_proba, multi_class='ovr')
```
#### Confusion Matrix
``` Python
cm_dt = confusion_matrix(YY_test,YY_pred_dt)
print("Confusion Matrix:")
print(cm_dt)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/dbb9f616-2685-44b4-8cea-c993bc579b98)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b00f071a-e3dc-4544-909d-fd0a273966cf)


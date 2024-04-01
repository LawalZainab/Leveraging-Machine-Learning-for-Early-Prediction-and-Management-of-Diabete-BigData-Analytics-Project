# Leveraging Machine Learning for Early Prediction and Management of Diabetes -  Big Data Analytics Project 

## Project Overview

Diabetes is a chronic metabolic disorder characterized by elevated blood sugar levels, resulting from insufficient insulin production or ineffective utilization. This disorder is known to be one of the most common chronic diseases affecting people around the globe. 
This research explores the application of machine learning in predicting diabetes, focusing on enhancing early detection and management. The outcomes aim to empower healthcare professionals with actionable insights, contributing to proactive healthcare strategies and improved patient outcomes. 


## Research Questions
- The main aim of this research is to propose a multiclass classification methodology for the prediction of diabetes.
  
This research focuses on two datasets from the laboratory of Medical City Hospital (the Specializes Center for Endocrinology and Diabetes-Al-Kindy Teaching Hospital) and another dataset from Vanderbilt, which is based on a study of rural African Americans in 
         Virginia.
1.  To employ different supervised multiclass classification algorithms that would accurately predict the onset of diabetes based on the predictor risk factors.
2.  What features are most influential in developing robust machine learning models for diabetes risk assessment?
3.  What machine learning technique is optimal to create a predictive model for diabetes?
4.  Evaluate and compare the developed system and performance of each algorithm.
5.  Comparing which dataset accurately predicts diabetes.

   
## Theme: 
Supervised Learning - Classification Algorithm

## Data Sources:
 This research paper mainly uses two datasets:
 
1- The Diabetes Dataset of 1000 Iraqi patients, acquired from the laboratory of Medical City Hospital and (the Specializes Center for Endocrinology and Diabetes-Al-Kindy Teaching Hospital). 

 [Diabetes Dataset - Mendeley Data](https://data.mendeley.com/datasets/wj9rwkp9c2/1)
 
2- The Diabetes Dataset from Vanderbilt, which is based on a study of rural African Americans in Virginia.

 https://data.world/informatics-edu/diabetes-prediction   Diabetes.csv
[[Diabetes Dataset - Mendeley Data](https://data.mendeley.com/datasets/wj9rwkp9c2/1)](https://data.mendeley.com/datasets/wj9rwkp9c2/1/files/2eb60cac-96b8-46ea-b971-6415e972afc9)

## Al-Kindy Diabetes Data Description

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/92540bb0-8c76-4e27-bcd2-196c0d4141a2)

## Vanderbilt Diabetes Data Description

   ![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f33de591-140d-42e0-8343-bc50f8b4cabf)

### Data Preperation: Vanderbilt Diabetes

-  Two(2) additional features(columns) has been added to the Vanderbilt datasets: CLASS and BMI
 
  1. CLASS - Using glycohemoglobin readings, Diabetic Screening Guidelines & Rules were used to classify patients as diabetic, pre-diabetic, or non-diabetic. Diabetic Screening Guidelines & Rules states that 
 
       - Non-diabetes  patients have  = below 5.7% glycohemoglobin readings
       - Pre-diabetes patients have  = 5.7 and 6.4% glycohemoglobin readings
       - Diabetes patients have = above 6.4% glycohemoglobin readings
   
  2.  BMI  was calculated using  =  703 x weight (lbs)/ [height(inches]2  
  
      - Weight- 2 sets of Weight data was provided in the original datasets : Weight1 and Weigh2,  The avergae weights was used in the calcaulation of BMI   

## Techniques:
- Machine Learning Algorithms that will be used are:  Random Forest and Decision Tree.

## Relevant tools:
- Python- for data analysis and visual representation of the datasets.


### 1- Data Downloading and Inspection
   
Downloading datasets: datasets were downloaded from the open source websites


### Libraries Used
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
from sklearn.ensemble import GradientBoostingClassifier
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from scipy.stats import randint
from sklearn.metrics import brier_score_loss
```

### Storing the  Al-Kindy Diabetes dataset

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

Additionally the info also tells us that the column ID, No of Patients and Cr are of data type int, the column Gender and CLASS are of  data type object, the column Urea, HbA2c, Chol, TG, HDL,LDL, VLDL,BMI are data type float. 

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


###  Exploratory Data Analysis and Data Visualization - Al- Kindy Diabetes Datasets
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
data.duplicated().sum()
data['ID'].nunique()
data.shape[0] - data['ID'].nunique()
data.duplicated(subset = ['ID']).sum() # this tally with the number we counted above
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/6c9f5331-1f6a-4245-b77f-5165eaca625f)

### Removing the Non-biological features
Non-biological variables in the dataset were dropped ('ID', 'No_Patients') becuase it will not provide any information on patients classes.

``` Python
df = data.drop(['ID', 'No_Patients'], axis=1)
df
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/5c700a9e-30c9-4844-abf9-0d99e9d6ab5f)


### Checking Class count and Gender Count
This provide insights on the:
1. numbers of Nondiabetes, Prediabetes and Diabetes in the datasets
2. numbers of Males and Females in the dataset
   
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

###  Label encoding the Class 'N': 1, 'P' : 2, 'Y' : 3
Label encoding involves assigning an integer value to each categorical variable.
 
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


### Median value of Non-diabetes, Pre-diabetes and Diabetes
``` Python
df.groupby('CLASS').median()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/411cdf37-4b58-456b-bee9-691fed1fe092)

It was observed that the dataset has outliers, there is a High gap difference between the minimum values and the first quartile of the Age, Urea, Cr, Chol, HbA1c. Furthermore, the considerable discrepancy between the maximum values and the third quartile of the 
Age, Urea, Cr, HbA1c

``` Python
df.describe()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/5a9cf075-c0d1-4cae-9e3d-87b3a0632575)


![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/e1cb1586-c3d0-49e4-91b6-dfc709202791)


### Ydata profiling 2 was performed 
Another exploratory analysis was performed, as the dataset has reduced from 1000 rows to 800 rows.

``` Python
profile_df = ProfileReport(df)
profile_df
```
From the 2nd Ydata profiling above, it shows that we have duplicates row. Note we have dealt with duplicate previously using the patients ID, the duplcate row could be patients have similar features. Hence, nothing will be done
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b5cec27d-3a9a-45ec-af89-a84b4fd5f915)

This shows the numbers of duplicated row
``` Python
dups = df.duplicated()
print('Number of duplicate row = %d' % (dups.sum()))
df[dups]
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/00784dd2-0ec3-46f8-b502-468efbd0ee2d)

``` Python
dups.shape
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b20dc562-48d6-4b42-97aa-9320ffc48ec8)

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
### Onehot ecoding Gender
One hot encoding:  was performed on the feature ‘Gender’ because it is a categorical variable containing label (Males and Females) values rather than numeric values. One hot encoding is performed because machine learning algorithms cannot operate on label data directly. They require all input variables and output variables to be numeric.

``` Python
X = pd.get_dummies(X, dtype='int')
X.head()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/d721eb3d-a636-46ce-b1b0-be0840c86393)


``` Python
print(Y)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/25cb1be2-4693-4f7f-8841-9681ec8f9244)


``` Python
df['CLASS'].value_counts()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ff32a0c9-b6c2-4132-88e6-e12f3048b39c)

### Class Distribution Before Resampling( Pie Chart and Count Plot)

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


## Models Evaluation
To evaluate models, data frame will be split into training (XX_train, YY_train) and testing (XX_test, YY_test) sets.
Before the application of the Machine Learning Algorithm, the listed observed issues will be treated:
1-	Balancing of data frame
2-	Outlier treatment
3-	Feature scaling of the data frame

###  Splitting datasets
Splitting the data into two parts, the first part contains 80% of the data whereas the second part of the data contains the remaining 20% of the data. We do this to avoid over-fitting, the two parts are called training and test splits, which gives us a better idea as to how our algorithm performed during the testing phase.The training split gets 80% of the data and the test split has 20% of the data.
1.  Stratify = Y, this data-splitting strategy that ensures that the proportion of each class in the training and test sets is the same as that in the original dataset.Stratified sampling helps to ensure that the model is trained and evaluated on a representative 
    sample of the data, and it can improve the model's overall performance.
2. Random = 11, This ensure the reproducibility of our results
 
``` Python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, stratify= Y,  shuffle =True, random_state= 11)
```


### Balancing dataset
Balancing the dataset: it was observed that the dataset was imbalanced from the exploratory data analysis carried out. Several balancing techniques were carried out.
1.	Random Under sampling Technique.
2.	SMOTE technique.
3.	Combination of SMOTE and Tomek Link Technique.


``` Python
def plot_resampling_results(Y_resampled, title):
  plt.figure(figsize = (10, 4))
  pd.Series(Y_resampled).value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'Green'])
  plt.title(title)
  plt.show()
``` 

### Random Undersampling
Random undersampling involves randomly selecting examples from the majority class to delete from the training dataset.
This has the effect of reducing the number of examples in the majority class in the transformed version of the training dataset. This process can be repeated until the desired class distribution is achieved, such as an equal number of examples for each class.
This approach may be more suitable for those datasets where there is a class imbalance although a sufficient number of examples in the minority class, such a useful model can be fit.

``` Python
rus = RandomUnderSampler(random_state =101)
X_rus, Y_rus = rus.fit_resample(X_train, Y_train)
plot_resampling_results(Y_rus, 'Class Distribution After Random Undersampling')
``` 

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/146c5e29-8761-431a-a2ad-9f34cc616b06)

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
556 records were moved from the data, when random undersampling was applied

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a2121dac-c4d8-4a11-819d-150b35153812)

#### Outliers Treatment
``` Python
plt.figure(figsize = (15, 15))
X_rus[[ 'AGE', 'Urea', 'Cr','HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender_F', 'Gender_M']].boxplot(vert =0)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a764d93d-b6b6-402c-a281-79c57dcd1ea4)

``` Python
def  replace_outlier(col):
  Q1r, Q3r =np.quantile(col, [.25, .75])
  IQRr =Q3r -Q1r
  LLr =Q1r -1.5*IQRr
  ULr = Q3r + 1.5*IQRr
  return LLr, ULr # Winsorization -UL - Capping, LL - Flooring

df_numr = X_rus[[  'AGE', 'Urea', 'Cr','HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender_F', 'Gender_M']]

for i in df_numr.columns:
  LLr, ULr = replace_outlier(df_numr[i])
  df_numr[i] = np.where(df_numr[i]> ULr, ULr, df_numr[i])
  df_numr[i] = np.where(df_numr[i] < LLr, LLr, df_numr[i])  # Winsorization - Capping and Flooring

plt.figure(figsize = (15, 10))
df_numr.boxplot(vert=0)


```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/dde363c9-2687-41a7-9537-541d759ae39a)

``` Python
plt.figure(figsize= (7,7))
sns.set(font_scale = 0.5)
sns.heatmap(df_numr.corr(), annot =True)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/251ea0b4-c95b-42fb-a43c-121f2c5ca604)

``` Python
scalerrr = MinMaxScaler().fit(df_numr)
print(scalerrr)
scalerrr.transform(df_numr)

X_rus_scaled = scalerrr.transform(df_numr)
print(X_rus_scaled)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f63c0537-dc2b-4dcd-87dd-fff06fd32744)

#### Gradient Boosting Machines
``` Python
gbmr = GradientBoostingClassifier()
gbmr.fit(X_rus_scaled, Y_rus)
feature_importancesr = gbmr.feature_importances_
print(feature_importancesr)
indices_gb_r = np.argsort(feature_importancesr)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(X_rus_scaled.shape[1]),
feature_importancesr[indices_gb_r],
align ='center')

feat_labels = X_rus.columns
plt.xticks(range(X_rus_scaled.shape[1]),
           feat_labels[indices_gb_r], rotation = 90)
plt.xlim([-1,X_rus_scaled.shape[1]])
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/528c045c-9b6a-4c44-85d6-063e42e00f4a)
#### Cross-Validation
``` Python

cv_scores_gb_r = cross_val_score(gbmr, X_rus_scaled, Y_rus, cv=5)
print("Cross-validation Scores:", cv_scores_gb_r)
mean_cv_score_gb_r = cv_scores_gb_r.mean()
std_cv_score_gb_r = cv_scores_gb_r.std()
print("Mean Cross-validation Score:", mean_cv_score_gb_r)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_gb_r)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_gb_r) + 1), cv_scores_gb_r, marker='o', linestyle='-')
plt.title('Cross-validation Scores- Random Undersampling')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_gb_r) + 1))
plt.grid(True)
plt.show()
```
#### Confusion Matrix
``` Python

Y_pred_gb_r = gbmr.predict(X_test)
conf_matrix_gb_r = confusion_matrix(Y_test, Y_pred_gb_r)
conf_matrix_gb_r
cm_disp_gb_r = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_gb_r, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_gb_r.plot()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/833e1b81-ee57-4427-ae59-e68c26ac322d)


#### Calculate effectiveness metrics
accuracy_gb_r = np.mean(cv_scores_gb_r)
precision_gb_r = np.mean(cross_val_score(gbmr, X_rus_scaled, Y_rus, cv=5, scoring='precision_weighted'))
recall_gb_r = np.mean(cross_val_score(gbmr,X_rus_scaled, Y_rus, cv=5, scoring='recall_weighted'))
f1_gb_r = np.mean(cross_val_score(gbmr,X_rus_scaled, Y_rus, cv=5, scoring='f1_weighted'))

# Print results
``` Python
#print("Brier Score:", brier_score_r)
print("Accuracy:", accuracy_gb_r)
print("Precision:", precision_gb_r)
print("Recall:", recall_gb_r)
print("F1 Score:", f1_gb_r)
``` 

#### Measure efficiency (training time and inference speed)
``` Python
start_time_gb_r = time.time()
gbmr.fit(X_rus_scaled, Y_rus)
end_time_gb_r= time.time()
training_time_gb_r = end_time_gb_r - start_time_gb_r

inference_start_time_gb_r = time.time()
Y_pred_inference_gb_r = gbmr.predict(X_test)
inference_end_time_gb_r = time.time()
inference_time_gb_r = inference_end_time_gb_r - inference_start_time_gb_r

print(f"Training Time: {training_time_gb_r:.4f} seconds")
print(f"Inference Speed: {inference_time_gb_r:.4f} seconds per prediction")
```





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

1058 records were added to the dataset when SMOTE oversampling technique was applied

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8525e62e-3ebd-4199-8548-fbaf6da52650)

``` Python
plt.figure(figsize = (15, 15))
X_smote[[ 'AGE', 'Urea', 'Cr','HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender_F', 'Gender_M']].boxplot(vert =0)

``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ad4d53e4-8fa6-48bb-b107-ec76fc51c11d)

``` Python
def  replace_outlier(col):
  Q1sm, Q3sm =np.quantile(col, [.25, .75])
  IQRsm =Q3sm -Q1sm
  LLsm =Q1sm -1.5*IQRsm
  ULsm = Q3sm + 1.5*IQRsm
  return LLsm, ULsm # Winsorization -UL - Capping, LL - Flooring

df_numsm = X_smote[[  'AGE', 'Urea', 'Cr','HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender_F', 'Gender_M']]

for i in df_numsm.columns:
  LLsm, ULsm = replace_outlier(df_numsm[i])
  df_numsm[i] = np.where(df_numsm[i]> ULsm, ULsm, df_numsm[i])
  df_numsm[i] = np.where(df_numsm[i] < LLsm, LLsm, df_numsm[i])  # Winsorization - Capping and Flooring

plt.figure(figsize = (15, 10))
df_numsm.boxplot(vert=0)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ed3cf142-b083-4a4a-9893-adc19d24dd2f)
``` Python
plt.figure(figsize= (7,7))
sns.set(font_scale = 0.5)
sns.heatmap(df_numsm.corr(), annot =True)
``` 

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/7b2bc1cc-37c7-4543-b7ba-f53716742b8f)

``` Python
scalersm = MinMaxScaler().fit(df_numsm)
print(scalersm)
scalersm.transform(df_numsm)
X_smote_scaled = scalersm.transform(df_numsm)
print(X_smote_scaled)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/889a984f-9196-4ffd-9b40-930d58746ab2)

### GradientBoostingClassifier - SMOTE

``` Python
gbmsm = GradientBoostingClassifier()
gbmsm.fit(X_smote_scaled, Y_smote)
feature_importancessm = gbmsm.feature_importances_
print(feature_importancessm)
indices_gb_sm = np.argsort(feature_importancessm)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(X_smote_scaled.shape[1]),
feature_importancessm[indices_gb_sm],
align ='center')

feat_labels = X_smote.columns
plt.xticks(range(X_smote_scaled.shape[1]),
           feat_labels[indices_gb_sm], rotation = 90)
plt.xlim([-1,X_smote_scaled.shape[1]])



```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f941d842-c202-4aa9-8af8-8acebc657940)
#### Cross-Validation
``` Python

cv_scores_gb_sm = cross_val_score(gbmsm, X_smote_scaled, Y_smote, cv=5)
print("Cross-validation Scores:", cv_scores_gb_sm)

mean_cv_score_gb_sm = cv_scores_gb_sm.mean()
std_cv_score_gb_sm = cv_scores_gb_sm.std()
print("Mean Cross-validation Score:", mean_cv_score_gb_sm)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_gb_sm)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_gb_sm) + 1), cv_scores_gb_sm, marker='o', linestyle='-')
plt.title('Cross-validation Scores- Smote')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_gb_sm) + 1))
plt.grid(True)
plt.show()

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/91abd427-4ba2-467e-9ff0-de5f9c685a96)

``` Python
# Confusion Matrix
Y_pred_gbsm = gbmsm.predict(X_test)
conf_matrix_gbsm = confusion_matrix(Y_test, Y_pred_gbsm)
conf_matrix_gbsm
cm_disp_gb_sm = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_gbsm, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_gb_sm.plot()

``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/803da8ef-c86d-438f-af48-3bf03599706e)

``` Python
report_gb_sm = classification_report(Y_test,Y_pred_gbsm, labels=[1,2,3],  target_names=["Nondiabetes", "Prediabetes", "Diabetes"])
print(report_gb_sm)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a1b48d39-d004-404f-96a8-14f6b0091c41)

``` Python
# Calculate effectiveness metrics
accuracy_gb_sm = np.mean(cv_scores_gb_sm)
precision_gb_sm = np.mean(cross_val_score(gbmsm, X_smote_scaled, Y_smote, cv=5, scoring='precision_weighted'))
recall_gb_sm = np.mean(cross_val_score(gbmsm,X_smote_scaled, Y_smote, cv=5, scoring='recall_weighted'))
f1_gb_sm = np.mean(cross_val_score(gbmsm,X_smote_scaled, Y_smote, cv=5, scoring='f1_weighted'))

# Print results
#print("Brier Score:", brier_score_gb)
print("Accuracy:", accuracy_gb_sm)
print("Precision:", precision_gb_sm)
print("Recall:", recall_gb_sm)
print("F1 Score:", f1_gb_sm)
``` 

``` Python
# Measure efficiency (training time and inference speed)
start_time_gb_sm = time.time()
gbmsm.fit(X_smote_scaled, Y_smote)
end_time_gb_sm= time.time()
training_time_gb_sm = end_time_gb_sm - start_time_gb_sm

inference_start_time_gb_sm = time.time()
Y_pred_inference_gb_sm = gbmsm.predict(X_test)
inference_end_time_gb_sm = time.time()
inference_time_gb_sm = inference_end_time_gb_sm - inference_start_time_gb_sm

print(f"Training Time: {training_time_gb_sm:.4f} seconds")
print(f"Inference Speed: {inference_time_gb_sm:.4f} seconds per prediction")
``` 

### Combination of SMOTE and Tomek Links Technique
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
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3eea3548-53cd-45e2-bf69-45567c182642)

``` Python
Y_st.value_counts()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3059866d-1a86-455f-9b0a-0d5ac3fdf1a8)

``` Python
print('No. of records added:', Y_st.shape[0] - Y_train.shape[0])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/4a812901-96a9-43a2-b89a-e2a17e797e91)

1056 were added to the datasets when SMOTET technique was applied.

### checking number of cells in the X_st dataframe.
X_st = X dataframe when SMOTET was applied

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/4a71a9d5-7889-4b15-bdcb-80ab719a4e9e)

Note: X_st was selected out of the 3 techniques as it employed both undersampling and oversampling

### Plotting Boxplot to visualize the outliers present in the dataframe( X_st)

#### Approaches to identify outliers and influential observations
Box plots identify interesting data points, or outliers. Its  a graphical approach that displays the distribution of data and indicates which observations might be outliers. Points that are beyond 1.5 times the IQR are beyond the expected range of variation of the data.
When analyzing data, identifying and addressing outliers is crucial. These anomalies can skew results, leading to inaccurate insights and decisions.

``` Python
plt.figure(figsize = (10, 10))
X_st[['AGE', 'Urea', 'Cr','HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']].boxplot(vert =0)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f9f12bf3-437f-485b-b7ec-f26d4f09f8b8)

### Treating the Outliers
If a value is higher than the 1.5*IQR above the upper quartile (Q3), the value will be considered as outlier. Similarly, if a value is lower than the 1.5*IQR below the lower quartile (Q1), the value will be considered as outlier.
QR is interquartile range. It measures dispersion or variation. IQR = Q3 -Q1.
Lower limit of acceptable range = Q1 - 1.5* (Q3-Q1)
Upper limit of acceptable range = Q3 + 1.5* (Q3-Q1)

Outliers were handled by employing Winsorisation technique. This process  of replace a specified number of extreme values with a smaller data value  involve chnaging the values of the ouliers. This is done to limit the effect of outliers or abnormal extreme values, or outliers, on the calculation.
 
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

### Plotting Boxplot to visualize the dataframe after treating the Outliers
``` Python
plt.figure(figsize = (15, 10))
df_num.boxplot(vert=0)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8e4315b7-e088-4193-9ca6-175f5c173be4)


### Plotting the Correlation heatmap to see the relatonship between the features
A correlation heatmap is a visual graphic that shows how each variable in the dataset are correlated to one another. -1 signifies zero correlation, while 1 signifies a perfect correlation. Correlation heatmaps are important because it helps identify which variables may potentially result in multicolinarity, which would compromise the integrity of the model. Multicolinearity happens when two or more features in a model are correlated with one another


``` Python
plt.figure(figsize= (10,10))
sns.set(font_scale = 1.0)
sns.heatmap(df_num.corr(), annot =True)
```

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3d630016-3412-415d-876c-7405555c0e7e)

It was observed  that
1- 	TG and VLDL are highly positively correlated
2- 	BMI and HbA1c are highly positively correlated

The embedded feature selection in the model technique will be used in selecting the most important features, so we dont have worry about the correlated features.

### Feature Scaling using Min Max Scaling
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
1.	handling missing values (missing data) in the dataset.
2.	producing lower error i.e. improves performance by reducing variance.
3.	effectively handling large amounts of training data efficiently i.e. resistant to irrelevant features.
4.	providing good classification results and avoiding overfitting

### Traning the model 
We define the parameters for the random forest training as follows:
1. n_estimators: This is the number of trees in the random forest classification. We have defined 500 trees in our random forest.
2. criterion: This is the loss function used to measure the quality of the split. There are two available options in sklearn — gini and entropy. We have used entropy.
3. random_state: This is the seed used by the random state generator for randomizing the dataset.

Next, we use the training dataset (both dependent and independent to train the random forest), as well as checking the important features.

``` Python
fr = RandomForestClassifier(n_estimators=500, criterion = 'entropy',  random_state =11)
fr.fit(X_st_scaled, Y_st)
importances = fr.feature_importances_
print(importances)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/1badcd17-af80-46dd-92c5-ca83db8009f3)

Argsort function was used  to perform an indirect sort along the given axis using the algorithm specified by the kind keyword,  It returns an array of indices of the same shape as arr that would sort the array.  It means indices of value arranged in ascending order.
Next was plotting the important features in ascending order. 

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
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b097df7a-a201-4138-8d03-d16bcad8c3dd)

## Evaluatiing the Performance

### Prediction
Performance evaluation of the trained model consists of following steps:
1. Predicting the species class of the test data using test feature set (X_st). We will use the predict function of the random forest classifier to predict classes.
2. Evaluating the performance of the classifier using Accuracy, Precision, Recall, Confusion Matrix, ROC Curve, F1 Score


``` Python
Y_pred_fr = fr.predict(X_test)
Y_pred_fr
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/2755b19e-19c8-4c5a-b286-0effd4448f09)


We'll be using "classification_report" to measure the quality of our predictions for each algorithm 
Now let's define some of the terms in "classification_report" that will be used to analyze each model
1. Accuracy: Fraction (or percentage) of predictions that were correct.
2. Precision: Fraction (or percentage) of correct predictions among all examples predicted to be positive, meaning what percent of our predictions were correct.
3. Recall: Fraction (or percentage) of correct predictions among all real positive examples. In simple terms, what percent of the positive cases did we catch properly.
4. F1-Score: Weighted harmonic mean of precision and recall. In simple terms, what percent of positive predictions were correct.

``` Python
print(classification_report(Y_test, Y_pred_fr))
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a1fb7026-338a-4dd7-8c96-953c49737890)


### Confusion Matrix

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

### f1 score
``` Python
f1_ddt = f1_score(Y_test,Y_pred_ddt, average= 'weighted')
print(f'F1 Score:{f1:.2f}')
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ac90fc43-ad28-4c1f-a400-7ccb10120585)



### Confusion matrix
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

### Data  Observations
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


### Checking for duplicate
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

### Checking the Class count and Gender Count
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
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/85b63d33-c66e-4f03-8cb1-97077509bb88)

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

``` Python
print(YY)
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/46626e93-9587-4f1f-9679-b0d7d82c6c82)

## Model Evaluation
To evaluate models, dataset was split into training (XX_train, YY_train) and testing (XX_test, YY_test) sets.
Before the application of the Machine Learning Algorithm, the listed observed issues will be treated
1-	Onehot encoding of variable (Gender)
2-	Handling missing cells 
3-	Balancing of dataset 
4-	Outliers’ treatment
5-	feature scaling of dataset 

### Onehot - creating seperate column for female and male

``` Python
XX = pd.get_dummies(XX, dtype ='int')
XX.head()
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/9fefca42-e412-44c6-8ec3-7f4aa9a84f39)


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
XX_train, XX_test, YY_train, YY_test = train_test_split(XX, YY, test_size=0.30, stratify= YY,shuffle =True, random_state= 11)
 ```
## Handling missing cells

Following the split of the data into training and test sets, it was discovered that some cells were missing. To address this issue, a condition was applied to the split which resulted in the missing cells being moved from the training set to the testing set. This condition was necessary to ensure that the missing cells were handled correctly during the training process, and to prevent any errors from occurring during prediction.

### Checking missing cells in training set and test set
``` Python
XX_train.isnull().sum()
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/99a27315-6757-430c-a91d-21a01506b258)
``` Python
XX_test.isnull().sum()
 ```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/82ba4b52-f702-4629-a031-b7c2a5efe3d6)

#### Identify missing cell in testing set
``` Python
missing_cells_mask =  XX_test.isnull().any(axis =1)
 ```
#### Moving missing cells to training sets
``` Python
XX_train_missing = XX_test[missing_cells_mask]
YY_train_missing = YY_test[XX_train_missing.index]
``` 
#### Removing rows with missing values from the testing set
``` Python
XX_test = XX_test.dropna()
YY_test = YY_test.loc[XX_test.index]
 ```
#### Concatenate the training set with the row containing missing values
``` Python
XX_train = pd.concat([XX_train, XX_train_missing])
YY_train = pd.concat([YY_train, YY_train_missing])
```
#### Confirming all missing cells are in training sets
``` Python
XX_train.isnull().sum()
XX_test.isnull().sum()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/68c54125-9b02-4aaa-8507-3561e0b55c8f)

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/03ff071c-e56a-415e-9c5b-e279e8a44a25)

### Confirming test set does not have missing cells
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


### Balancing the training sets

Balancing the dataset: it was observed that the dataset was imbalanced from the exploratory data analysis carried out. Several balancing techniques were carried out.
1-	Random Under sampling Technique.
2-	SMOTE technique.
3-	Combination of SMOTE and Tomek Link Technique.



``` Python
def plot_resampling_results(YY_resampled, title):
  plt.figure(figsize = (10, 4))
  pd.Series(YY_resampled).value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'Green'])
  plt.title(title)
  plt.show()
``` 
### Technique 1:  Random Undersampling Vanderbilt Datasets
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

### Box plot under random undersampling
``` Python
plt.figure(figsize = (15, 15))
XX_russ[[ 'chol', 'stab.glu', 'hdl','ratio', 'glyhb', 'Age', 'Height', 'Weight', 'BMI', 'Systolic_Blood_Pressure', 'Diastolic_Blood_Pressure','waist', 'hip', 'Gender_female', 'Gender_male']].boxplot(vert =0)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/661cb2e5-2ee6-4648-a8e0-7d9498984121)
``` Python
def  replace_outlier(col):
  Q1r, Q3r =np.quantile(col, [.25, .75])
  IQRr =Q3r -Q1r
  LLr =Q1r -1.5*IQRr
  ULr = Q3r + 1.5*IQRr
  return LLr, ULr # Winsorization -UL - Capping, LL - Flooring

  df_numr = XX_russ[[ 'chol', 'stab.glu', 'hdl','ratio', 'glyhb', 'Age', 'Height', 'Weight', 'BMI', 'Systolic_Blood_Pressure', 'Diastolic_Blood_Pressure','waist', 'hip', 'Gender_female', 'Gender_male']]

for i in df_numr.columns:
  LLr, ULr = replace_outlier(df_numr[i])
  df_numr[i] = np.where(df_numr[i]> ULr, ULr, df_numr[i])
  df_numr[i] = np.where(df_numr[i] < LLr, LLr, df_numr[i])  # Winsorization - Capping and Flooring

  plt.figure(figsize = (15, 10))
df_numr.boxplot(vert=0)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/c42e211b-a5f2-4037-9bf5-32bda10a6e7c)
``` Python
plt.figure(figsize= (7,7))
sns.set(font_scale = 0.5)
sns.heatmap(df_numr.corr(), annot =True)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/e7549068-6751-4c00-a2bc-ad78c9964045)
``` Python
scalerrr = MinMaxScaler().fit(df_numr)
print(scalerrr)
scalerrr.transform(df_numr)
XX_russ_scaled = scalerrr.transform(df_numr)
print(XX_russ_scaled)
```
### Gradient Boosting Machine

``` Python
gbmr = GradientBoostingClassifier()
gbmr.fit(XX_russ_scaled, YY_russ)
feature_importancesr = gbmr.feature_importances_
indices_gb_r = np.argsort(feature_importancesr)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_russ_scaled.shape[1]),
feature_importancesr[indices_gb_r],
align ='center')

feat_labels = XX_russ.columns
plt.xticks(range(XX_russ_scaled.shape[1]),
           feat_labels[indices_gb_r], rotation = 90)
plt.xlim([-1,XX_russ_scaled.shape[1]])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/25e33889-c619-4fed-a8c1-5a37a6314760)

####  Cross-Validation
``` Python
cv_scores_gb_r = cross_val_score(gbmr, XX_russ_scaled, YY_russ, cv=5)
print("Cross-validation Scores:", cv_scores_gb_r)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3eb183a0-25b7-43ea-b08a-e900f1c0e6e9)
``` Python
mean_cv_score_gb_r = cv_scores_gb_r.mean()
std_cv_score_gb_r = cv_scores_gb_r.std()
print("Mean Cross-validation Score:", mean_cv_score_gb_r)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_gb_r)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a7c74ba0-fce0-4b3b-b139-65fabf4b0b81)
``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_gb_r) + 1), cv_scores_gb_r, marker='o', linestyle='-')
plt.title('Cross-validation Scores- Random Undersampling')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_gb_r) + 1))
plt.grid(True)
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/af1e1a04-d8ad-4b6f-89a4-d80f6bdfe057)

#### Confusion Matrix
``` Python
YY_pred_gb_r = gbmr.predict(XX_test)
conf_matrix_gb_r = confusion_matrix(YY_test, YY_pred_gb_r)
cm_disp_gb_r = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_gb_r, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_gb_r.plot()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/7af8a944-3726-4ea0-821e-01fb63a82824)

#### Calculate effectiveness metrics
``` Python
accuracy_gb_r = np.mean(cv_scores_gb_r)
precision_gb_r = np.mean(cross_val_score(gbmr, XX_russ_scaled, YY_russ, cv=5, scoring='precision_weighted'))
recall_gb_r = np.mean(cross_val_score(gbmr,XX_russ_scaled, YY_russ, cv=5, scoring='recall_weighted'))
f1_gb_r = np.mean(cross_val_score(gbmr,XX_russ_scaled, YY_russ, cv=5, scoring='f1_weighted'))

# Print results
#print("Brier Score:", brier_score_r)
print("Accuracy:", accuracy_gb_r)
print("Precision:", precision_gb_r)
print("Recall:", recall_gb_r)
print("F1 Score:", f1_gb_r)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/936f8aff-2ce6-4ffb-8044-34a371b3b22c)

#### Measure efficiency (training time and inference speed)
``` Python
start_time_gb_r = time.time()
gbmr.fit(XX_russ_scaled, YY_russ)
end_time_gb_r= time.time()
training_time_gb_r = end_time_gb_r - start_time_gb_r

inference_start_time_gb_r = time.time()
YY_pred_inference_gb_r = gbmr.predict(XX_test)
inference_end_time_gb_r = time.time()
inference_time_gb_r = inference_end_time_gb_r - inference_start_time_gb_r

print(f"Training Time: {training_time_gb_r:.4f} seconds")
print(f"Inference Speed: {inference_time_gb_r:.4f} seconds per prediction")
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/4d99ae41-683c-4b3c-b9d7-f16e244c977b)

### Technique 2: SMOTE( Synthetic Minority Over-Sampling Technique)- Vanderbilt Datasets
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
``` Python
plt.figure(figsize = (15, 15))
XX_smote[[ 'chol', 'stab.glu', 'hdl','ratio', 'glyhb', 'Age', 'Height', 'Weight', 'BMI', 'Systolic_Blood_Pressure', 'Diastolic_Blood_Pressure','waist', 'hip', 'Gender_female', 'Gender_male']].boxplot(vert =0)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/7df2d16e-d56c-41d8-a58f-7c1b48306c81)
``` Python
def  replace_outlier(col):
  Q1s, Q3s =np.quantile(col, [.25, .75])
  IQRs =Q3s -Q1s
  LLs =Q1s -1.5*IQRs
  ULs = Q3s + 1.5*IQRs
  return LLs, ULs # Winsorization -UL - Capping, LL - Flooring

df_nums = XX_smote[[ 'chol', 'stab.glu', 'hdl','ratio', 'glyhb', 'Age', 'Height', 'Weight', 'BMI', 'Systolic_Blood_Pressure', 'Diastolic_Blood_Pressure','waist', 'hip', 'Gender_female', 'Gender_male']]

for i in df_nums.columns:
  LLs, ULs = replace_outlier(df_nums[i])
  df_nums[i] = np.where(df_nums[i]> ULs, ULs, df_nums[i])
  df_nums[i] = np.where(df_nums[i] < LLs, LLs, df_nums[i])  # Winsorization - Capping and Flooring

plt.figure(figsize = (15, 10))
df_nums.boxplot(vert=0)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/fa7ddfec-4c88-4b75-a82e-eae93e78811e)
``` Python
plt.figure(figsize= (7,7))
sns.set(font_scale = 0.5)
sns.heatmap(df_nums.corr(), annot =True)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/cf3e3f44-1a61-4c02-893a-d34bf11f59c5)
``` Python
scalers = MinMaxScaler().fit(df_nums)
print(scalers)
scalers.transform(df_nums)
XX_smote_scaled = scalers.transform(df_nums)
print(XX_smote_scaled)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3c313065-65f9-46c0-901e-61d0a39f8cb2)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/14ab28b8-b059-46db-832b-b2e71ccf0580)

``` Python
gbms = GradientBoostingClassifier()
gbms.fit(XX_smote_scaled, YY_smote)
feature_importancess = gbms.feature_importances_
print(feature_importancess)
indices_gb_s = np.argsort(feature_importancess)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_smote_scaled.shape[1]),
feature_importancess[indices_gb_s],
align ='center')

feat_labels = XX_smote.columns
plt.xticks(range(XX_smote_scaled.shape[1]),
           feat_labels[indices_gb_s], rotation = 90)
plt.xlim([-1,XX_smote_scaled.shape[1]])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8fca8610-e3e7-4347-be71-0ab8b9f59ea6)

#### Cross-Validation
``` Python
cv_scores_gb_s = cross_val_score(gbms, XX_smote_scaled, YY_smote, cv=5)
print("Cross-validation Scores:", cv_scores_gb_s)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/d64b266f-2bfc-4544-9f38-bec49a7109c2)
``` Python
mean_cv_score_gb_s = cv_scores_gb_s.mean()
std_cv_score_gb_s = cv_scores_gb_s.std()
print("Mean Cross-validation Score:", mean_cv_score_gb_s)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_gb_s)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/2a83c652-2580-46af-97a2-d9da530b5ab1)
``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_gb_s) + 1), cv_scores_gb_s, marker='o', linestyle='-')
plt.title('Cross-validation Scores- Smote')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_gb_s) + 1))
plt.grid(True)
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/9e0ef098-3585-4288-92b3-7c34872a12d0)

#### Confusion Matrix
``` Python
YY_pred_gbs = gbms.predict(XX_test)
conf_matrix_gbs = confusion_matrix(YY_test, YY_pred_gbs)
conf_matrix_gbs 
cm_disp_gb_s = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_gbs, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_gb_s.plot()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/c64c6aa0-865a-4a68-a59e-e2d1d46e8b9d)
``` Python
report_gb_s = classification_report(YY_test,YY_pred_gbs, labels=[1,2,3],  target_names=["Nondiabetes", "Prediabetes", "Diabetes"])
print(report_gb_s)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ee29d089-342d-47e3-ba9b-39d008d534f1)

#### Calculate effectiveness metrics
``` Python
accuracy_gb_s = np.mean(cv_scores_gb_s)
precision_gb_s = np.mean(cross_val_score(gbms, XX_smote_scaled, YY_smote, cv=5, scoring='precision_weighted'))
recall_gb_s = np.mean(cross_val_score(gbms,XX_smote_scaled, YY_smote, cv=5, scoring='recall_weighted'))
f1_gb_s = np.mean(cross_val_score(gbms,XX_smote_scaled, YY_smote, cv=5, scoring='f1_weighted'))
``` 
#### Print results
``` Python
#print("Brier Score:", brier_score_gb)
print("Accuracy:", accuracy_gb_s)
print("Precision:", precision_gb_s)
print("Recall:", recall_gb_s)
print("F1 Score:", f1_gb_s)
```
#### Measure efficiency (training time and inference speed)
``` Python

start_time_gb_s = time.time()
gbms.fit(XX_smote_scaled, YY_smote)
end_time_gb_s= time.time()
training_time_gb_s = end_time_gb_s - start_time_gb_s

inference_start_time_gb_s = time.time()
Y_pred_inference_gb_s = gbms.predict(XX_test)
inference_end_time_gb_s = time.time()
inference_time_gb_s = inference_end_time_gb_s - inference_start_time_gb_s

print(f"Training Time: {training_time_gb_s:.4f} seconds")
print(f"Inference Speed: {inference_time_gb_s:.4f} seconds per prediction")
```

### Technique 3: Combination of SMOTE and Tomek Links
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
``` Python
plt.figure(figsize = (15, 15))
XX_st[[ 'chol', 'stab.glu', 'hdl','ratio', 'glyhb', 'Age', 'Height', 'Weight', 'BMI', 'Systolic_Blood_Pressure', 'Diastolic_Blood_Pressure','waist', 'hip', 'Gender_female', 'Gender_male']].boxplot(vert =0)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8e2800e7-88ad-4735-b375-51fb16b1c329)


``` Python
def  replace_outlier(col):
  Q1st, Q3st =np.quantile(col, [.25, .75])
  IQRst =Q3st -Q1st
  LLst =Q1st -1.5*IQRst
  ULst = Q3st + 1.5*IQRst
  return LLst, ULst # Winsorization -UL - Capping, LL - Flooring

df_numst = XX_st[[ 'chol', 'stab.glu', 'hdl','ratio', 'glyhb', 'Age', 'Height', 'Weight', 'BMI', 'Systolic_Blood_Pressure', 'Diastolic_Blood_Pressure','waist', 'hip', 'Gender_female', 'Gender_male']]

for i in df_numst.columns:
  LLst, ULst = replace_outlier(df_numst[i])
  df_numst[i] = np.where(df_numst[i]> ULst, ULst, df_numst[i])
  df_numst[i] = np.where(df_numst[i] < LLst, LLst, df_numst[i])  # Winsorization - Capping and Flooring

plt.figure(figsize = (15, 10))
df_numst.boxplot(vert=0)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/33488ce3-31c7-4fec-b1f9-1b3e610814f4)

``` Python
plt.figure(figsize= (7,7))
sns.set(font_scale = 0.5)
sns.heatmap(df_numst.corr(), annot =True)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/55af98e0-a4a2-4dc8-bbf9-c07ef0283cbb)

``` Python
scalerst = MinMaxScaler().fit(df_numst)
print(scalerst)

scalerst.transform(df_numst)
XX_st_scaled = scalerst.transform(df_numst)
print(XX_st_scaled)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f45f81ed-ff5d-4793-8355-2f2b178b5c56)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ea2f9396-3495-4a28-8261-5cc23cc33252)

``` Python
gbmst = GradientBoostingClassifier()
gbmst.fit(XX_st_scaled, YY_st)
feature_importances_st = gbmst.feature_importances_
print(feature_importances_st)
indices_gb_st = np.argsort(feature_importances_st)[::-1]

plt.ylabel('Feature importance')
plt.bar(range(XX_st_scaled.shape[1]),
feature_importances_st[indices_gb_st],
align ='center')

feat_labels = XX_st.columns
plt.xticks(range(XX_st_scaled.shape[1]),
           feat_labels[indices_gb_st], rotation = 90)
plt.xlim([-1,XX_st_scaled.shape[1]])

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/1a5b1d1c-02c0-4b8f-8b60-7bb15b3429a5)
#### Cross-Validation
``` Python

cv_scores_gb_st = cross_val_score(gbmst, XX_st_scaled, YY_st, cv=5)
print("Cross-validation Scores:", cv_scores_gb_st)
``` 

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/16a1e154-7308-444f-9111-4e3c5db00f2f)
``` Python

mean_cv_score_gb_st = cv_scores_gb_st.mean()
std_cv_score_gb_st = cv_scores_gb_st.std()
print("Mean Cross-validation Score:", mean_cv_score_gb_st)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_gb_st)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3bd7fb6c-29a6-48af-a1b7-1505d3473be2)

``` Python

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_gb_st) + 1), cv_scores_gb_st, marker='o', linestyle='-')
plt.title('Cross-validation Scores- Smote +Tomek  Gradient Boosting Machine')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_gb_st) + 1))
plt.grid(True)
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0cced37e-b756-405c-a665-c2034f9b7cd5)

#### Confusion Matrix
``` Python

YY_pred_gb_st = gbmst.predict(XX_test)
conf_matrix_gb_st = confusion_matrix(YY_test, YY_pred_gb_st)

cm_disp_gb_st = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_gb_st, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_gb_st.plot()

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/6c828847-e7b8-4ad5-a9f2-e122c3874b76)
#### Calculate effectiveness metrics
``` Python
 
accuracy_gb_st = np.mean(cv_scores_gb_st)
precision_gb_st = np.mean(cross_val_score(gbmst, XX_st_scaled, YY_st, cv=5, scoring='precision_weighted'))
recall_gb_st = np.mean(cross_val_score(gbmst,XX_st_scaled, YY_st, cv=5, scoring='recall_weighted'))
f1_gb_st = np.mean(cross_val_score(gbmst,XX_st_scaled, YY_st, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_gb_st)
print("Precision:", precision_gb_st)
print("Recall:", recall_gb_st)
print("F1 Score:", f1_gb_st)

```
``` Python
reportst = classification_report(YY_test,YY_pred_gb_st, labels=[1,2,3],  target_names=["Nondiabetes", "Prediabetes", "Diabetes"])
print(reportst)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/1ae34fe7-60bb-4348-9bac-9c888b0367ba)

``` Python

# Measure efficiency (training time and inference speed)
start_time_gb_st = time.time()
gbmst.fit(XX_st_scaled, YY_st)
end_time_gb_st= time.time()
training_time_gb_st = end_time_gb_st - start_time_gb_st

inference_start_time_gb_st = time.time()
YY_pred_inference_gb_st = gbmst.predict(XX_test)
inference_end_time_gb_st = time.time()
inference_time_gb_st = inference_end_time_gb_st- inference_start_time_gb_st

print(f"Training Time: {training_time_gb_st:.4f} seconds")
print(f"Inference Speed: {inference_time_gb_st:.4f} seconds per prediction")
```
## Random Forest
Random Forest: can be defined as a collection of tree-type classifiers. It uses simple probability to select the strongest features for its inputs. The advantages of RF include:
1-	handling missing values (missing data) in the dataset.
2-	producing lower error i.e. improves performance by reducing variance.
3-	effectively handling large amounts of training data efficiently i.e. resistant to irrelevant features.
4-	providing good classification results and avoiding overfitting

### Random undersampling on Random forest - Entropy
``` Python

fr =RandomForestClassifier(n_estimators=50,  criterion = 'entropy', random_state =11)
fr.fit(XX_russ_scaled, YY_russ)
importancesr = fr.feature_importances_
print(importancesr)

indicesr = np.argsort(importancesr)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_russ_scaled.shape[1]),
importancesr[indicesr],
align ='center')

feat_labels = XX_russ.columns
plt.xticks(range(XX_russ_scaled.shape[1]),
           feat_labels[indicesr], rotation = 90)
plt.xlim([-1,XX_russ_scaled.shape[1]])

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/62d81468-0a24-46ad-98a7-b4b82a105710)
#### Cross-Validation
``` Python

cv_scores_fr = cross_val_score(fr, XX_russ_scaled, YY_russ, cv=5)
print("Cross-validation Scores:", cv_scores_fr)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3824d58a-22f0-42fa-b43d-57d8312fb125)
``` Python
mean_cv_score_fr = cv_scores_fr.mean()
std_cv_score_fr = cv_scores_fr.std()
print("Mean Cross-validation Score:", mean_cv_score_fr)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_fr)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/c7c3a673-9712-47f0-8f3c-7b8de677063e)
``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_fr) + 1), cv_scores_fr, marker='o', linestyle='-')
plt.title('Cross-validation Scores-Random Forest- Entropy')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_fr) + 1))
plt.grid(True)
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0bada448-c465-44d3-a120-3ee3c6afc104)


#### Confusion Matrix
``` Python
YY_pred_fr = fr.predict(XX_test)
conf_matrix_fr = confusion_matrix(YY_test, YY_pred_fr)

cm_disp_fr = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_fr, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_fr.plot()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/d4700b6b-e7b2-4dcb-a1ff-64763d104f15)

``` Python
 # Calculate effectiveness metrics
accuracy_fr = np.mean(cv_scores_fr)
precision_fr = np.mean(cross_val_score(fr, XX_russ_scaled, YY_russ, cv=5, scoring='precision_weighted'))
recall_fr = np.mean(cross_val_score(fr, XX_russ_scaled, YY_russ, cv=5, scoring='recall_weighted'))
f1_fr = np.mean(cross_val_score(fr,XX_russ_scaled, YY_russ, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_fr)
print("Precision:", precision_fr)
print("Recall:", recall_fr)
print("F1 Score:", f1_fr)
``` 

``` Python
report_fr = classification_report(YY_test,YY_pred_fr, labels=[1,2,3],  target_names=["Nondiabetes", "Prediabetes", "Diabetes"])
print(report_fr)
```
#### Measure efficiency (training time and inference speed)
``` Python

start_time_fr = time.time()
fr.fit(XX_russ_scaled, YY_russ)
end_time_fr= time.time()
training_time_fr = end_time_fr - start_time_fr

inference_start_time_fr = time.time()
YY_pred_inference_fr = fr.predict(XX_test)
inference_end_time_fr = time.time()
inference_time_fr = inference_end_time_fr- inference_start_time_fr

print(f"Training Time: {training_time_fr:.4f} seconds")
print(f"Inference Speed: {inference_time_fr:.4f} seconds per prediction")
```
``` Python
for i in range(3):
    tree = fr.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=XX_st.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/6323d857-f87e-48b6-9c58-9f3bb40bdbdf)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8f36cbe0-d17e-446f-a1ec-154b134c59ac)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/19f28a46-766e-4d86-9eca-5b6e7dfbc698)


### Random undersampling on Random forest - Gini

``` Python
fr1 =RandomForestClassifier(n_estimators=50,  criterion = 'gini', random_state =11)
fr1.fit(XX_russ_scaled, YY_russ)
importancessr = fr1.feature_importances_
print(importancessr)

indices_sr = np.argsort(importancessr)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_russ_scaled.shape[1]),
importancessr[indices_sr],
align ='center')

feat_labels = XX_russ.columns
plt.xticks(range(XX_russ_scaled.shape[1]),
           feat_labels[indices_sr], rotation = 90)
plt.xlim([-1,XX_russ_scaled.shape[1]])

``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/04a282af-20df-4737-a942-86cb9b49e65c)

#### Cross-Validation

``` Python
cv_scores_fr1 = cross_val_score(fr1, XX_russ_scaled, YY_russ, cv=5)
print("Cross-validation Scores:", cv_scores_fr1)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/22ad0584-da10-4979-a46a-3b8e7617e462)


``` Python
mean_cv_score_fr1 = cv_scores_fr1.mean()
std_cv_score_fr1 = cv_scores_fr1.std()
print("Mean Cross-validation Score:", mean_cv_score_fr1)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_fr1)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/5d48e1dc-6a0e-43cf-ab79-96172b751f61)

``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_fr1) + 1), cv_scores_fr1, marker='o', linestyle='-')
plt.title('Cross-validation Scores-Random Forest-Gini')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_fr1) + 1))
plt.grid(True)
plt.show()

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/5d2c46cf-5f1a-41a7-8883-b2b3e8a36f67)

#### Confusion Matrix
``` Python
YY_pred_fr1 = fr1.predict(XX_test)
conf_matrix_fr1 = confusion_matrix(YY_test, YY_pred_fr1)

cm_disp_fr1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_fr1, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_fr1.plot()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/aece0da2-7900-4273-8cbb-5c47ecb4efc3)

### Plotting Boxplot to visualize the outliers present in the dataframe

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/79dc8abe-2d8c-42b0-82e9-599b0e8fa465)
#### Calculate effectiveness metrics
``` Python
 
accuracy_fr1 = np.mean(cv_scores_fr1)
precision_fr1 = np.mean(cross_val_score(fr1, XX_russ_scaled, YY_russ, cv=5, scoring='precision_weighted'))
recall_fr1 = np.mean(cross_val_score(fr1, XX_russ_scaled, YY_russ, cv=5, scoring='recall_weighted'))
f1_fr1 = np.mean(cross_val_score(fr1,XX_russ_scaled, YY_russ, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_fr1)
print("Precision:", precision_fr1)
print("Recall:", recall_fr1)
print("F1 Score:", f1_fr1)
``` 

``` Python
report_fr1 = classification_report(YY_test,YY_pred_fr1, labels=[1,2,3],  target_names=["Nondiabetes", "Prediabetes", "Diabetes"])
print(report_fr1)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f4df75af-48e4-4308-ba51-db22dce13e28)

``` Python
# Measure efficiency (training time and inference speed)
start_time_fr1 = time.time()
fr1.fit(XX_russ_scaled, YY_russ)
end_time_fr1= time.time()
training_time_fr1 = end_time_fr1 - start_time_fr1

inference_start_time_fr1 = time.time()
y_pred_inference_fr1 = fr1.predict(XX_test)
inference_end_time_fr1 = time.time()
inference_time_fr1 = inference_end_time_fr1- inference_start_time_fr1

print(f"Training Time: {training_time_fr1:.4f} seconds")
print(f"Inference Speed: {inference_time_fr1:.4f} seconds per prediction")

``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/4ba8e736-2aca-444a-a6bd-0dbf109a7f87)

``` Python
for i in range(3):
    tree1 = fr1.estimators_[i]
    dot_data1 = export_graphviz(tree1,
                               feature_names=XX_russ.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph1 = graphviz.Source(dot_data1)
    display(graph1)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/60ca6568-8b91-4a5f-b4a7-e814f3520b40)

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/cc4cd7b5-4ed5-46d6-9695-23c576d97322)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/50152cca-4a79-4ab2-89bd-a57a6e60d7ae)

### SMOTE on Random forest - Entropy

``` Python
frs =RandomForestClassifier(n_estimators=50,  criterion = 'entropy',  random_state =11)
frs.fit(XX_smote_scaled, YY_smote)
importancessmt = frs.feature_importances_
print(importancessmt)
indicessmt = np.argsort(importancessmt)[::-1]

plt.ylabel('Feature importance')
plt.bar(range(XX_smote_scaled.shape[1]),
importancessmt[indicessmt],
align ='center')

feat_labels = XX_smote.columns
plt.xticks(range(XX_smote_scaled.shape[1]),
           feat_labels[indicessmt], rotation = 90)
plt.xlim([-1,XX_smote_scaled.shape[1]])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/c3d554db-9705-45d2-b109-c0f9073ee91f)

#### Cross-Validation
``` Python
cv_scores_frs = cross_val_score(frs, XX_smote_scaled, YY_smote, cv=5)
print("Cross-validation Scores:", cv_scores_frs)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0915a8a9-c2e0-4026-81db-9af9145842ca)

``` Python
mean_cv_score_frs = cv_scores_frs.mean()
std_cv_score_frs = cv_scores_frs.std()
print("Mean Cross-validation Score:", mean_cv_score_frs)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_frs)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/d2e005de-9567-4d23-a441-14771ef57127)

``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_frs) + 1), cv_scores_frs, marker='o', linestyle='-')
plt.title('Cross-validation Scores')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_frs) + 1))
plt.grid(True)
plt.show()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b148666b-6ec9-4507-ad23-f3ddcafee8a2)


#### Confusion Matrix
``` Python
YY_pred_frs = frs.predict(XX_test)
conf_matrix_frs = confusion_matrix(YY_test, YY_pred_frs)

cm_disp_frs = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_frs, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_frs.plot()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3d23ec0a-fef9-4276-88b4-be433111b8f7)


``` Python
report_frs = classification_report(YY_test,YY_pred_frs, labels=[1,2,3],  target_names=["Nondiabetes", "Prediabetes", "Diabetes"])
print(report_frs)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/50b0dea3-70a4-48bc-a4c4-e0565b05dcd1)

#### Calculate effectiveness metrics
``` Python
accuracy_frs = np.mean(cv_scores_frs)
precision_frs = np.mean(cross_val_score(frs, XX_smote_scaled, YY_smote, cv=5, scoring='precision_weighted'))
recall_frs = np.mean(cross_val_score(frs, XX_smote_scaled, YY_smote, cv=5, scoring='recall_weighted'))
f1_frs = np.mean(cross_val_score(frs,XX_smote_scaled, YY_smote, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_frs)
print("Precision:", precision_frs)
print("Recall:", recall_frs)
print("F1 Score:", f1_frs)
``` 
#### Measure efficiency (training time and inference speed)
``` Python
start_time_frs = time.time()
frs.fit(XX_smote_scaled, YY_smote)
end_time_frs = time.time()
training_time_frs = end_time_frs - start_time_frs

inference_start_time_frs = time.time()
y_pred_inference_frs = frs.predict(XX_test)
inference_end_time_frs = time.time()
inference_time_frs = inference_end_time_frs - inference_start_time_frs


print(f"Training Time: {training_time_frs:.4f} seconds")
print(f"Inference Speed: {inference_time_frs:.4f} seconds per prediction")
```

### SMOTE on Random forest - Gini
``` Python
fr2 =RandomForestClassifier(n_estimators=50,  criterion = 'gini',  random_state =11)
fr2.fit(XX_smote_scaled, YY_smote)
importances_2 = fr2.feature_importances_

print(importances_2)

indices_2 = np.argsort(importances_2)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_smote_scaled.shape[1]),
importances_2[indices_2],
align ='center')

feat_labels = XX_smote.columns
plt.xticks(range(XX_smote_scaled.shape[1]),
           feat_labels[indices_2], rotation = 90)
plt.xlim([-1,XX_smote_scaled.shape[1]])

``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/73cc3e73-5246-4389-b6a7-f9b96f173f6e)


#### Cross-Validation
``` Python
cv_scores_fr2 = cross_val_score(fr2, XX_smote_scaled, YY_smote, cv=5)
print("Cross-validation Scores:", cv_scores_fr2)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/e090d714-7a93-4ed0-a6b4-0894b47701ec)

``` Python
mean_cv_score_fr2 = cv_scores_fr2.mean()
std_cv_score_fr2 = cv_scores_fr2.std()
print("Mean Cross-validation Score:", mean_cv_score_fr2)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_fr2)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/5032a12e-c033-4fb2-9093-5e28a8d1ba30)

``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_fr2) + 1), cv_scores_fr2, marker='o', linestyle='-')
plt.title('Cross-validation Scores')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_fr2) + 1))
plt.grid(True)
plt.show()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/fa774e08-133a-45d6-9555-656528a7f0c2)

#### Confusion Matrix
``` Python
YY_pred_fr2 = fr2.predict(XX_test)
conf_matrix_fr2 = confusion_matrix(YY_test, YY_pred_fr2)

cm_disp_fr2 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_fr2, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_fr2.plot()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/582770d9-2b4e-4da4-a68b-e9154a1e2115)

``` Python
report_fr2 = classification_report(YY_test,YY_pred_fr2, labels=[1,2,3],  target_names=["Nondiabetes", "Prediabetes", "Diabetes"])
print(report_fr2)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/8c4fe8f6-d6a9-44f3-8f74-3695dce8d6a5)


#### Calculate effectiveness metrics
 ``` Python
accuracy_fr2 = np.mean(cv_scores_fr2)
precision_fr2 = np.mean(cross_val_score(fr2, XX_smote_scaled, YY_smote, cv=5, scoring='precision_weighted'))
recall_fr2 = np.mean(cross_val_score(fr2, XX_smote_scaled, YY_smote, cv=5, scoring='recall_weighted'))
f1_fr2 = np.mean(cross_val_score(fr2,XX_smote_scaled, YY_smote, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_fr2)
print("Precision:", precision_fr2)
print("Recall:", recall_fr2)
print("F1 Score:", f1_fr2)
``` 
#### Measure efficiency (training time and inference speed)
``` Python
start_time_fr2 = time.time()
fr2.fit(XX_smote_scaled, YY_smote)
end_time_fr2 = time.time()
training_time_fr2 = end_time_fr2 - start_time_fr2

inference_start_time_fr2 = time.time()
y_pred_inference_fr2 = fr2.predict(XX_test)
inference_end_time_fr2 = time.time()
inference_time_fr2 = inference_end_time_fr2 - inference_start_time_fr2

print(f"Training Time: {training_time_fr2:.4f} seconds")
print(f"Inference Speed: {inference_time_fr2:.4f} seconds per prediction")
```
``` Python
for i in range(3):
    tree2 = fr2.estimators_[i]
    dot_data2 = export_graphviz(tree2,
                               feature_names=XX_st.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph2 = graphviz.Source(dot_data2)
    display(graph2)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/dd4b0a30-75a0-4666-be4c-a79582220ec3)

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/56cd0a8d-d886-4f59-866f-0da0b41c05b9)

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f9b291f8-fe89-4710-bce3-bc1deb020be9)


### SMOTE + Tomek on Random forest - Entropy

``` Python
fr3 =RandomForestClassifier(n_estimators=50,  criterion = 'entropy', random_state =11)

fr3.fit(XX_st_scaled, YY_st)
importancesfrst = fr3.feature_importances_
print(importancesfrst)

indicesfrst = np.argsort(importancesfrst)[::-1]

plt.ylabel('Feature importance')
plt.bar(range(XX_st_scaled.shape[1]),
importancesfrst[indicesfrst],
align ='center')

feat_labels = XX.columns
plt.xticks(range(XX_st_scaled.shape[1]),
           feat_labels[indicesfrst], rotation = 90)
plt.xlim([-1,XX_st_scaled.shape[1]])
``` 

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ef40d0d7-77c3-4b93-a241-5b0233e113f0)

#### Cross-Validation
``` Python

cv_scores_fr3 = cross_val_score(fr3, XX_st_scaled, YY_st, cv=5)
print("Cross-validation Scores:", cv_scores_fr3)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3fd7a9be-dcdc-48eb-8684-41a90cab2b01)

``` Python
mean_cv_score_fr3 = cv_scores_fr3.mean()
std_cv_score_fr3 = cv_scores_fr3.std()
print("Mean Cross-validation Score:", mean_cv_score_fr3)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_fr3)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/e3ca3845-fe64-4e33-9d24-a7700c441426)


``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_fr3) + 1), cv_scores_fr3, marker='o', linestyle='-')
plt.title('Cross-validation Scores Random Forest Smote + Tomek')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_fr3) + 1))
plt.grid(True)
plt.show()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3074f431-6696-4592-87b8-477f75e184a9)

#### Confusion Matrix


``` Python
YY_pred_fr3 = fr3.predict(XX_test)
conf_matrix_fr3 = confusion_matrix(YY_test, YY_pred_fr3)

cm_disp_fr3 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_fr3, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_fr3.plot()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0d9fd47d-4ea1-44db-a5f6-c6c182ad8348)

``` Python
report_fr3 = classification_report(YY_test,YY_pred_fr3, labels=[1,2,3],  target_names=["Nondiabetes", "Prediabetes", "Diabetes"])
print(report_fr3)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/31cb4b7a-b6f8-4b89-aef4-51bb7a827596)

#### Calculate effectiveness metrics
``` Python

accuracy_fr3 = np.mean(cv_scores_fr3)
precision_fr3 = np.mean(cross_val_score(fr3, XX_st_scaled, YY_st, cv=5, scoring='precision_weighted'))
recall_fr3 = np.mean(cross_val_score(fr3, XX_st_scaled, YY_st, cv=5, scoring='recall_weighted'))
f1_fr3 = np.mean(cross_val_score(fr3,XX_st_scaled, YY_st, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_fr3)
print("Precision:", precision_fr3)
print("Recall:", recall_fr3)
print("F1 Score:", f1_fr3)

```

#### Measure efficiency (training time and inference speed)

``` Python
start_time_fr3 = time.time()
fr3.fit(XX_st_scaled, YY_st)
end_time_fr3 = time.time()
training_time_fr3 = end_time_fr3 - start_time_fr3

inference_start_time_fr3 = time.time()
y_pred_inference_fr3 = fr3.predict(XX_test)
inference_end_time_fr3 = time.time()
inference_time_fr3 = inference_end_time_fr3 - inference_start_time_fr3

print(f"Training Time: {training_time_fr3:.4f} seconds")
print(f"Inference Speed: {inference_time_fr3:.4f} seconds per prediction")

```
#### Visualizing the first three trees
``` Python
for i in range(3):
    tree3 = fr3.estimators_[i]
    dot_data3 = export_graphviz(tree3,
                               feature_names=XX_st.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph3 = graphviz.Source(dot_data3)
    display(graph3)
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0db773fd-9765-4fb7-af3b-1c05b43a3948)

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/c2a0e308-8448-4f68-a2ef-6c95d715560b)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/b3d477d0-1347-4720-ac88-ec8246d63626)

#### SMOTE + Tomek on Random forest - Gini
``` Python
fr4 =RandomForestClassifier(n_estimators=50,  criterion = 'gini',  random_state =11)
fr4.fit(XX_st_scaled, YY_st)
importances_4 = fr4.feature_importances_
print(importances_4)

indices_4 = np.argsort(importances_4)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_st_scaled.shape[1]),
importances_4[indices_4],
align ='center')

feat_labels = XX_st.columns
plt.xticks(range(XX_st_scaled.shape[1]),
           feat_labels[indices_4], rotation = 90)
plt.xlim([-1,XX_st_scaled.shape[1]])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/7d3eb1e7-6f61-4fa3-99ce-668301ec3b40)

#### Cross-Validation
``` Python

cv_scores_fr4 = cross_val_score(fr4, XX_st_scaled, YY_st, cv=5)
print("Cross-validation Scores:", cv_scores_fr4)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/5ac6e985-74f3-4711-bbf7-210ea9aa287b)

``` Python
mean_cv_score_fr4 = cv_scores_fr4.mean()
std_cv_score_fr4 = cv_scores_fr4.std()
print("Mean Cross-validation Score:", mean_cv_score_fr4)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_fr4)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/ccec011e-636d-445f-84d1-c3d7bac7a340)

``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_fr4) + 1), cv_scores_fr4, marker='o', linestyle='-')
plt.title('Cross-validation Scores Smote + Tomek Random Forest')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_fr4) + 1))
plt.grid(True)
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/78f33db5-4f0c-4efd-a41a-1821641c40d2)

#### Confusion Matrix
``` Python
YY_pred_fr4 = fr4.predict(XX_test)
conf_matrix_fr4 = confusion_matrix(YY_test, YY_pred_fr4)

cm_disp_fr4 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_fr4, display_labels=["Nondiabetes", "Prediabetes", "Diabetes"])
cm_disp_fr4.plot()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/1e952a3b-abe1-41e5-ab0a-13001efe3a09)

``` Python
report_fr4 = classification_report(YY_test,YY_pred_fr4, labels=[1,2,3],  target_names=["Nondiabetes", "Prediabetes", "Diabetes"])
print(report_fr4)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/7a59bcbf-e7b3-47dd-b13a-fc7c5af5ef28)

``` Python
 # Calculate effectiveness metrics
accuracy_fr4 = np.mean(cv_scores_fr3)
precision_fr4 = np.mean(cross_val_score(fr4, XX_st_scaled, YY_st, cv=5, scoring='precision_weighted'))
recall_fr4 = np.mean(cross_val_score(fr4, XX_st_scaled, YY_st, cv=5, scoring='recall_weighted'))
f1_fr4 = np.mean(cross_val_score(fr4,XX_st_scaled, YY_st, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_fr4)
print("Precision:", precision_fr4)
print("Recall:", recall_fr4)
print("F1 Score:", f1_fr4)
``` 

#### Measure efficiency (training time and inference speed)
``` Python
start_time_fr4 = time.time()
fr4.fit(XX_st_scaled, YY_st)
end_time_fr4 = time.time()
training_time_fr4 = end_time_fr4 - start_time_fr4


inference_start_time_fr4 = time.time()
y_pred_inference_fr4 = fr4.predict(XX_test)
inference_end_time_fr4 = time.time()
inference_time_fr4 = inference_end_time_fr4 - inference_start_time_fr4

print(f"Training Time: {training_time_fr4:.4f} seconds")
print(f"Inference Speed: {inference_time_fr4:.4f} seconds per prediction")

```

#### Visualizing the first three trees
``` Python
for i in range(3):
    tree4 = fr4.estimators_[i]
    dot_data4 = export_graphviz(tree4,
                               feature_names=XX_st.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data4)
    display(graph)

``` 

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/5b4897bc-40f9-46f1-82f9-d7b3b7c4cf23)
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/96a59432-713d-481f-bc91-1074491871be)

![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0a4ea775-b51b-42e9-ab9e-53bbab3c8c1c)





## Decision Tree on Vanderbilt Diabetes Datasets

### Feature selection using the embedded technique- Decision Tree Classifier 
Decision Tree: is a non-parametric supervised learning algorithm for classification and regression tasks. That can handle numerical & categorical data and multi-output problems. Its

1-	validate a model using statistical tests. That makes it possible to account for the reliability of the model.
2-	performs well even if its assumptions are somewhat violated by the true model from which the data were generated.


#### Decision Trees Random UnderSampling Entropy
``` Python
dt = DecisionTreeClassifier(criterion = 'entropy',  random_state =42)
dt.fit(XX_russ_scaled, YY_russ)
importances_dt = dt.feature_importances_
print(importances_dt)

indices_dt = np.argsort(importances_dt)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_russ_scaled.shape[1]),
importances_dt[indices_dt],
align ='center')

feat_labels = XX_russ.columns
plt.xticks(range(XX_russ_scaled.shape[1]),
           feat_labels[indices_dt], rotation = 90)
plt.xlim([-1,XX_russ_scaled.shape[1]])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/c768bc71-e48e-446b-8d48-9ffc1d377144)

#### Cross Validation 
``` Python

cv_scores_dt = cross_val_score(dt, XX_russ_scaled, YY_russ, cv=5)
print("Cross-validation Scores:", cv_scores_dt)
``` 
``` Python
mean_cv_score_dt = cv_scores_dt.mean()
std_cv_score_dt = cv_scores_dt.std()
print("Mean Cross-validation Score:", mean_cv_score_dt)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_dt)
``` 
``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_dt) + 1), cv_scores_dt, marker='o', linestyle='-')
plt.title('Cross-validation Scores Decision Trees UnderSampling')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_dt) + 1))
plt.grid(True)
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0696b1e9-e6ca-4d63-92fe-9daa88454021)

``` Python
 # Calculate effectiveness metrics
accuracy_dt = np.mean(cv_scores_dt)
precision_dt = np.mean(cross_val_score(dt, XX_russ_scaled, YY_russ, cv=5, scoring='precision_weighted'))
recall_dt = np.mean(cross_val_score(dt, XX_russ_scaled, YY_russ, cv=5, scoring='recall_weighted'))
f1_dt = np.mean(cross_val_score(dt,XX_russ_scaled, YY_russ, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1 Score:", f1_dt)

``` 
``` Python
# Measure efficiency (training time and inference speed)
start_time_dt  = time.time()
dt .fit(XX_russ_scaled, YY_russ)
end_time_dt  = time.time()
training_time_dt  = end_time_dt - start_time_dt

inference_start_time_dt = time.time()
y_pred_inference_dt = dt.predict(XX_test)
inference_end_time_dt = time.time()
inference_time_dt = inference_end_time_dt- inference_start_time_dt

print(f"Training Time: {training_time_dt :.4f} seconds")
print(f"Inference Speed: {inference_time_dt:.4f} seconds per prediction")

```
#### Decision Trees Random UnderSampling Gini
``` Python
dtt = DecisionTreeClassifier(criterion = 'gini',  random_state =42)
dtt.fit(XX_russ_scaled, YY_russ)
importances_dtt = dt.feature_importances_

print(importances_dtt)

indices_dtt = np.argsort(importances_dt)[::-1]

plt.ylabel('Feature importance')
plt.bar(range(XX_russ_scaled.shape[1]),
importances_dtt[indices_dt],
align ='center')

feat_labels = XX_russ.columns
plt.xticks(range(XX_russ_scaled.shape[1]),
           feat_labels[indices_dt], rotation = 90)
plt.xlim([-1,XX_russ_scaled.shape[1]])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/1101a5bd-e431-429a-a07e-bd573baa3079)

#### Cross Validation
``` Python
cv_scores_dtt = cross_val_score(dtt, XX_russ_scaled, YY_russ, cv=5)
print("Cross-validation Scores:", cv_scores_dtt)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/051a998b-afa4-4f43-b046-85e320b5077a)

``` Python
mean_cv_score_dtt = cv_scores_dtt.mean()
std_cv_score_dtt = cv_scores_dtt.std()
print("Mean Cross-validation Score:", mean_cv_score_dtt)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_dtt)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f83de22a-ddb4-47ae-8cd2-431c00565f56)

``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_dtt) + 1), cv_scores_dtt, marker='o', linestyle='-')
plt.title('Cross-validation Scores Decision Trees UnderSampling Gini')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_dtt) + 1))
plt.grid(True)
plt.show()
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/7eaaa68b-cb9c-4122-9f1f-4dff1288c18e)


#### Calculate effectiveness metrics
``` Python
accuracy_dtt = np.mean(cv_scores_dtt)
precision_dtt = np.mean(cross_val_score(dtt, XX_russ_scaled, YY_russ, cv=5, scoring='precision_weighted'))
recall_dtt = np.mean(cross_val_score(dtt, XX_russ_scaled, YY_russ, cv=5, scoring='recall_weighted'))
f1_dtt = np.mean(cross_val_score(dtt,XX_russ_scaled, YY_russ, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_dtt)
print("Precision:", precision_dtt)
print("Recall:", recall_dtt)
print("F1 Score:", f1_dtt)
```

#### Measure efficiency (training time and inference speed)
``` Python
start_time_dtt  = time.time()
dtt .fit(XX_russ_scaled, YY_russ)
end_time_dtt  = time.time()
training_time_dtt  = end_time_dtt - start_time_dtt

inference_start_time_dtt = time.time()
y_pred_inference_dtt = dtt.predict(XX_test)
inference_end_time_dtt = time.time()
inference_time_dtt = inference_end_time_dtt- inference_start_time_dtt

print(f"Training Time: {training_time_dtt :.4f} seconds")
print(f"Inference Speed: {inference_time_dtt:.4f} seconds per prediction")
```

#### Decision Trees SMOTE Entropy
``` Python
dsmt = DecisionTreeClassifier(criterion = 'entropy',  random_state =42)
dsmt.fit(XX_smote_scaled, YY_smote)
importances_dsmt = dsmt.feature_importances_

print(importances_dsmt )
indices_dsmt = np.argsort(importances_dsmt)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_smote_scaled.shape[1]),
importances_dsmt[indices_dsmt],
align ='center')

feat_labels = XX_smote.columns
plt.xticks(range(XX_smote_scaled.shape[1]),
           feat_labels[indices_dsmt], rotation = 90)
plt.xlim([-1,XX_smote_scaled.shape[1]])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/d1e8a8b8-bed1-4b32-b429-fa4818d13e9c)

 #### Cross Validation
``` Python
cv_scores_dsmt = cross_val_score(dsmt, XX_smote_scaled, YY_smote, cv=5)
print("Cross-validation Scores:", cv_scores_dsmt)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/519aa082-0f0b-4375-a216-dc5174a301be)
``` Python
mean_cv_score_dsmt = cv_scores_dsmt.mean()
std_cv_score_dsmt = cv_scores_dsmt.std()
print("Mean Cross-validation Score:", mean_cv_score_dsmt)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_dsmt)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/9e30b986-c364-4a2c-ac06-74569fc43d05)


``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_dsmt) + 1), cv_scores_dsmt, marker='o', linestyle='-')
plt.title('Cross-validation Scores')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_dsmt) + 1))
plt.grid(True)
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/6170ac27-9895-48a8-a29f-61be8c2f3a25)

#### Calculate effectiveness metrics
``` Python
accuracy_dsmt = np.mean(cv_scores_dsmt)
precision_dsmt = np.mean(cross_val_score(dsmt, XX_smote_scaled, YY_smote, cv=5, scoring='precision_weighted'))
recall_dsmt = np.mean(cross_val_score(dsmt, XX_smote_scaled, YY_smote, cv=5, scoring='recall_weighted'))
f1_dsmt = np.mean(cross_val_score(dsmt,XX_smote_scaled, YY_smote, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_dsmt)
print("Precision:", precision_dsmt)
print("Recall:", recall_dsmt)
print("F1 Score:", f1_dsmt)
```

#### Measure efficiency (training time and inference speed)
``` Python
start_time_dsmt  = time.time()
dsmt .fit(XX_smote_scaled, YY_smote)
end_time_dsmt  = time.time()
training_time_dsmt  = end_time_dsmt - start_time_dsmt

inference_start_time_dsmt = time.time()
y_pred_inference_dsmt = dsmt.predict(XX_test)
inference_end_time_dsmt = time.time()
inference_time_dsmt= inference_end_time_dsmt- inference_start_time_dsmt

print(f"Training Time: {training_time_dsmt :.4f} seconds")
print(f"Inference Speed: {inference_time_dsmt:.4f} seconds per prediction")
```

#### Decision Trees SMOTE Gini

``` Python
dsm = DecisionTreeClassifier(criterion = 'gini',  random_state =42)
dsm.fit(XX_smote_scaled, YY_smote)
importances_dsm = dsm.feature_importances_

print(importances_dsm )

indices_dsm = np.argsort(importances_dsm)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_smote_scaled.shape[1]),
importances_dsm[indices_dsm],
align ='center')

feat_labels = XX_smote.columns
plt.xticks(range(XX_smote_scaled.shape[1]),
           feat_labels[indices_dsm], rotation = 90)
plt.xlim([-1,XX_smote_scaled.shape[1]])
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/3e9e3952-9896-43af-b988-2bcf8797d076)


#### Cross Validation
``` Python

cv_scores_dsm = cross_val_score(dsm, XX_smote_scaled, YY_smote, cv=5)
print("Cross-validation Scores:", cv_scores_dsm)
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a5cc6d8f-f9d6-4b89-9868-65f864266fe9)

``` Python
mean_cv_score_dsm = cv_scores_dsm.mean()
std_cv_score_dsm = cv_scores_dsm.std()
print("Mean Cross-validation Score:", mean_cv_score_dsm)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_dsm)
```
``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_dsm) + 1), cv_scores_dsmt, marker='o', linestyle='-')
plt.title('Cross-validation Scores')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_dsm) + 1))
plt.grid(True)
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/25cfc0fd-9216-494f-806d-e01bc2118636)


#### Calculate effectiveness metrics
``` Python
accuracy_dsm = np.mean(cv_scores_dsm)
precision_dsm = np.mean(cross_val_score(dsm, XX_smote_scaled, YY_smote, cv=5, scoring='precision_weighted'))
recall_dsm = np.mean(cross_val_score(dsm, XX_smote_scaled, YY_smote, cv=5, scoring='recall_weighted'))
f1_dsm = np.mean(cross_val_score(dsm,XX_smote_scaled, YY_smote, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_dsm)
print("Precision:", precision_dsm)
print("Recall:", recall_dsm)
print("F1 Score:", f1_dsm)
```

#### Measure efficiency (training time and inference speed)
``` Python
start_time_dsm  = time.time()
dsm .fit(XX_smote_scaled, YY_smote)
end_time_dsm = time.time()
training_time_dsm  = end_time_dsm - start_time_dsm

inference_start_time_dsm = time.time()
y_pred_inference_dsm = dsm.predict(XX_test)
inference_end_time_dsm = time.time()
inference_time_dsm= inference_end_time_dsm- inference_start_time_dsm

print(f"Training Time: {training_time_dsm :.4f} seconds")
print(f"Inference Speed: {inference_time_dsm:.4f} seconds per prediction")
```

#### Decision Trees SMOTE + Tomek Entropy

``` Python
ddt = DecisionTreeClassifier(criterion = 'entropy',  random_state =42)
ddt.fit(XX_st_scaled, YY_st)
importances_ddt = ddt.feature_importances_

print(importances_ddt)
indices_ddt = np.argsort(importances_ddt)[::-1]

plt.ylabel('Feature importance')
plt.bar(range(XX_st_scaled.shape[1]),
importances_ddt[indices_ddt],
align ='center')

feat_labels = XX_st.columns
plt.xticks(range(XX_st_scaled.shape[1]),
           feat_labels[indices_ddt], rotation = 90)
plt.xlim([-1,XX_st_scaled.shape[1]])


```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/c28372a3-5f9e-45ae-a482-9ad12f5f9b1d)
 #### Cross Validation
``` Python
cv_scores_ddt = cross_val_score(ddt, XX_st_scaled, YY_st, cv=5)
print("Cross-validation Scores:", cv_scores_ddt)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/a90e35b0-db8a-46f4-938b-53ade01bdd3b)

``` Python
mean_cv_score_ddt = cv_scores_ddt.mean()
std_cv_score_ddt = cv_scores_ddt.std()
print("Mean Cross-validation Score:", mean_cv_score_ddt)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_ddt)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/f83e5592-5b5a-4145-a9f5-1540eb624d68)

``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_ddt) + 1), cv_scores_ddt, marker='o', linestyle='-')
plt.title('Cross-validation Scores')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_ddt) + 1))
plt.grid(True)
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/d32b9995-73f8-43db-8328-cb4b2d20f74c)

#### Calculate effectiveness metrics
``` Python
accuracy_ddt = np.mean(cv_scores_ddt)
precision_ddt = np.mean(cross_val_score(ddt, XX_st_scaled, YY_st, cv=5, scoring='precision_weighted'))
recall_ddt = np.mean(cross_val_score(ddt, XX_st_scaled, YY_st, cv=5, scoring='recall_weighted'))
f1_ddt = np.mean(cross_val_score(ddt,XX_st_scaled, YY_st, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_ddt)
print("Precision:", precision_ddt)
print("Recall:", recall_ddt)
print("F1 Score:", f1_ddt)
``` 
#### Measure efficiency (training time and inference speed)
``` Python
start_time_ddt  = time.time()
ddt.fit(XX_st_scaled, YY_st)
end_time_ddt = time.time()
training_time_ddt  = end_time_ddt- start_time_ddt

inference_start_time_ddt = time.time()
y_pred_inference_ddt = ddt.predict(XX_test)
inference_end_time_ddt = time.time()
inference_time_ddt= inference_end_time_ddt- inference_start_time_ddt

print(f"Training Time: {training_time_ddt :.4f} seconds")
print(f"Inference Speed: {inference_time_ddt:.4f} seconds per prediction")
``` 

#### Decision Trees SMOTE + Tomek Gini

``` Python
ddtt = DecisionTreeClassifier(criterion = 'gini',  random_state =42)
ddtt.fit(XX_st_scaled, YY_st)
importances_ddtt = ddtt.feature_importances_

print(importances_ddtt)
indices_ddtt = np.argsort(importances_ddtt)[::-1]
plt.ylabel('Feature importance')
plt.bar(range(XX_st_scaled.shape[1]),
importances_ddtt[indices_ddtt],
align ='center')

feat_labels = XX_st.columns
plt.xticks(range(XX_st_scaled.shape[1]),
           feat_labels[indices_ddtt], rotation = 90)
plt.xlim([-1,XX_st_scaled.shape[1]])
``` 
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/0837c7df-df50-487c-9cc7-6f756f73b942)

#### Cross Validation
``` Python
cv_scores_ddtt = cross_val_score(ddtt, XX_st_scaled, YY_st, cv=5)
print("Cross-validation Scores:", cv_scores_ddtt)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/eebbf168-f66c-4f86-bfe3-8dc6fe180cf2)
 
``` Python
mean_cv_score_ddtt = cv_scores_ddtt.mean()
std_cv_score_ddtt = cv_scores_ddtt.std()
print("Mean Cross-validation Score:", mean_cv_score_ddtt)
print("Standard Deviation of Cross-validation Scores:", std_cv_score_ddtt)

```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/227f4d42-1708-4635-953b-23fe512a8d0b)

``` Python
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores_ddtt) + 1), cv_scores_ddtt, marker='o', linestyle='-')
plt.title('Cross-validation Scores')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(range(1, len(cv_scores_ddtt) + 1))
plt.grid(True)
plt.show()
```
![image](https://github.com/LawalZainab/Leveraging-Machine-Learning-for-Early-Prediction-and-Management-of-Diabetes-BigDataAnalytics-Project/assets/157916270/07c1f4e4-d826-4930-9ccb-bc55a5112a3d)

 #### Calculate effectiveness metrics
 ``` Python
accuracy_ddtt = np.mean(cv_scores_ddtt)
precision_ddtt = np.mean(cross_val_score(ddtt, XX_st_scaled, YY_st, cv=5, scoring='precision_weighted'))
recall_ddtt = np.mean(cross_val_score(ddtt, XX_st_scaled, YY_st, cv=5, scoring='recall_weighted'))
f1_ddtt = np.mean(cross_val_score(ddtt,XX_st_scaled, YY_st, cv=5, scoring='f1_weighted'))

# Print results

#print("Brier Score:", brier_score_gb_st)

print("Accuracy:", accuracy_ddtt)
print("Precision:", precision_ddtt)
print("Recall:", recall_ddtt)
print("F1 Score:", f1_ddtt)
```

#### Measure efficiency (training time and inference speed)
``` Python
start_time_ddtt  = time.time()
ddtt.fit(XX_st_scaled, YY_st)
end_time_ddtt = time.time()
training_time_ddtt  = end_time_ddtt- start_time_ddtt


inference_start_time_ddtt = time.time()
y_pred_inference_ddtt = ddtt.predict(XX_test)
inference_end_time_ddtt = time.time()
inference_time_ddtt= inference_end_time_ddtt- inference_start_time_ddtt

print(f"Training Time: {training_time_ddtt :.4f} seconds")
print(f"Inference Speed: {inference_time_ddtt:.4f} seconds per prediction")
``` 


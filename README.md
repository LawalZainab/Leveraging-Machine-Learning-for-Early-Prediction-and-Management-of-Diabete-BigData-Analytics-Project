# Leveraging Machine Learning for Early Prediction and Management of Diabetes -  Big Data Analytics Project 

### Project Overview

Diabetes is a chronic metabolic disorder characterized by elevated blood sugar levels, resulting from insufficient insulin production or ineffective utilization. This disorder is known to be one of the most common chronic diseases affecting people around the globe. 
This research explores the application of machine learning in predicting diabetes, focusing on enhancing early detection and management. The outcomes aim to empower healthcare professionals with actionable insights, contributing to proactive healthcare strategies and improved patient outcomes. 

### Theme: 
Supervised Learning - Classification Algorithm

### Data Sources:
 This research paper mainly uses two datasets:
 
1- The Diabetes Dataset of 1000 Iraqi patients, acquired from the laboratory of Medical City Hospital and (the Specializes Center for Endocrinology and Diabetes-Al-Kindy Teaching Hospital). 

 The data consists of 11 variables on 1000 subjects:  Age, Gender, Creatinine ratio (Cr), Body Mass Index (BMI), Urea(High blood urea nitrogen), Cholesterol (Chol), LDL(low-density lipoprotein), VLDL(very low density lipoprotein), Triglycerides (TG), HDL Cholesterol(high-density lipoprotein), HBA1C(hemoglobin A1c), Class (the patient's diabetes disease class may be Diabetic, Non-Diabetic, or Pre-diabetic).

 [Diabetes Dataset - Mendeley Data](https://data.mendeley.com/datasets/wj9rwkp9c2/1)
       
2- The Diabetes Dataset from Vanderbilt, which is based on a study of rural African Americans in Virginia.

   The data consists of 19 variables on 390 subjects:  Patient ID, Chol(Cholesterol), stab.glu(Stabilized Glucose), HDL (high-density lipoprotein), ratio(Chol/hdl) gylhb(Glycated hemoglobin),Location, Age, Gender, Height, Weight 1, Weight 2, Height, Frame, bp.1s( first systolic blood pressure), bp.1d(first diastolic blood pressure), bp.2s( second systolic blood pressure), bp.2d (second diastolic blood pressure), waist, hip, time.ppn.
   
   https://data.world/informatics-edu/diabetes-prediction   Diabetes.csv
  
### Data Preperation:

- Vanderbilt dataset diagnosis of diabetes was based on glycohemoglobin readings. These readings have been grouped into Diabetes, Pre-diabetes, and Non-Diabetes based on the following ranges:
  
       - Non-diabetes = below 5.7%
       - Pre-diabetes = 5.7 and 6.4%
       - Diabetes = above 6.4%
   
- BMI was calculated 703 x weight (lbs)/ [height(inches]2
  
      - Weight- there were 2 sets of data for weight : Weight1 and Weigh2, We took the Avg Weight for the calculation of BMI


### Research Questions
- This research focuses on datasets from the laboratory of Medical City Hospital and (the Specializes Center for Endocrinology and Diabetes-Al-Kindy Teaching Hospital)
   - How can machine learning models contribute to early detection and accurate prediction of diabetes, considering diverse patient populations and datasets? Comparing the predictive performance of different Algorithm for early diabetes prediction
   - What features and variables are most influential in developing robust machine learning models for diabetes risk assessment
   - How can machine learning algorithms be optimized to accurately predict the onset of diabetes?
   
     
### Techniques:
- Machine Learning Algorithms that will be used are:  Decision Tree, Random Forest, K-Nearest Neighbors, Naive Bayes and Support Vector Machines.
  

### Relevant tools:
- Python- for data analysis and visual representation of the datasets.


#### 1- Data Downloading and Inspection
   
Downloading datasets:
[[Diabetes Dataset - Mendeley Data](https://data.mendeley.com/datasets/wj9rwkp9c2/1)](https://data.mendeley.com/datasets/wj9rwkp9c2/1/files/2eb60cac-96b8-46ea-b971-6415e972afc9)


#### 2- Exploratory Data Analysis

Dataset 1 - Al_Kindy
1. Install and Import the required packaged into Jupyter notebook.
2. Read and understanding  Dataset_of_Diabetes_Al-Kindy.csv 
3. View datasets
4. check data shape 
5. Check datatypes 
6. Check for missing values 
7. Check data summary: Minimum, Maximum, Mean, and the Percentiles (25%, 50%, and 75%) of the datasets.
8. Run EDA Ydata Profiling on the datasets.
9. Run EDA using Sweetviz for comparison with Ydata profiling

``` Python

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

Since there are 252 missing values  in bp.2s and bp.2d - we drop the variables, as bp.1s and bp.1d provided the systolic blood pressure and diastolic blood pressure needed for our analysis. 

``` Python
# To remove bp.2s and bp.2d from the dataframe
data3 = data2.drop(['bp.2s', 'bp.2d'], axis=1)
data3
profile_data3 = ProfileReport(data3)
profile_data3
```



# Leveraging Machine Learning for Early Prediction and Management of Diabetes -  Big Data Analytics Project 

### Project Overview

Diabetes is a chronic metabolic disorder characterized by elevated blood sugar levels, resulting from insufficient insulin production or ineffective utilization. This disorder is known to be one of the most common chronic diseases affecting people around the globe. 
This research explores the application of machine learning in predicting diabetes, focusing on enhancing early detection and management. The outcomes aim to empower healthcare professionals with actionable insights, contributing to proactive healthcare strategies and improved patient outcomes. 

### Theme: 
Supervised Learning - Classification Algorithm

### Data Sources:
 This research paper mainly uses two datsets:
 
1- The Diabetes Dataset of 1000 Iraqi patients, acquired from the laboratory of Medical City Hospital and (the Specializes Center for Endocrinology and Diabetes-Al-Kindy Teaching Hospital). 

 The data consists of 13 variables on 1000 subjects: No. of Patient, Sugar Level Blood, Age, Gender, Creatinine ratio (Cr), Body Mass Index (BMI), Urea, Cholesterol (Chol), Fasting lipid profile, including total, LDL(low-density lipoprotein), 
 VLDL(very low density lipoprotein), Triglycerides (TG), HDL Cholesterol(high-density lipoprotein), HBA1C(hemoglobin A1c), Class (the patient's diabetes disease class may be Diabetic, Non-Diabetic, or Predict-Diabetic).

 [Diabetes Dataset - Mendeley Data](https://data.mendeley.com/datasets/wj9rwkp9c2/1)
       
2- the Diabetes Dataset from Vanderbilt, which is based on a study of rural African Americans in Virginia

   The data consists of 19 variables on 390 subjects:  Chol(Cholesterol), stab.glu(Stabilized Glucose), hdl (high-density lipoprotein), ratio(Chol/hdl) gylhb(Glycated hemoglobin),Location, Age, Gender, height, Weight 1, Weight 2, Height, Frame, bp.1s( first systolic 
   blood pressure), bp.1d(first diastolic blood pressure), bp.2s( second systolic blood pressure), bp.2d (second diastolic blood pressure) waist, hip, time.ppn.

### Data Preperation:

- Vanderbilt dataset diagnosis of diabetes was based on a glycohemoglobin:
  
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
   -
     
### Techniques:
- Predictive models:  Decision tree, Random Forest, KNN, Naive Bayes, Random forest
  

### Relevant tools:
- Python- for data analysis and visual representation of the datasets.


#### 1- Data Downloading and Inspection
   
Downloading datasets:
[[Diabetes Dataset - Mendeley Data](https://data.mendeley.com/datasets/wj9rwkp9c2/1)](https://data.mendeley.com/datasets/wj9rwkp9c2/1/files/2eb60cac-96b8-46ea-b971-6415e972afc9)


#### 2- Exploratory Data Analysis


1. Install and Import the required packaged into Jupyter notebook.
2. Read and understanding  Dataset_of_Diabetes.csv 
3. View datasets
4. check data shape 
5. Check datatypes 
6. Check for missing values : none were detected.
7. Check data summary: Minimum, Maximum, Mean, and the Percentiles (25%, 50%, and 75%) of the datasets.
8. Run Ydata Profiling on the datasets.

``` Python

!pip install ydata-profiling
from ydata_profiling import ProfileReport
import pandas as pd
from google.colab import files
Dataset_of_Diabetes = files.upload()
data = pd.read_csv(r"Dataset_of_Diabetes.csv")
data
data.shape
data.info()
data.isnull().sum()
data.describe()
profile_data = ProfileReport(data)
profile_data
profile_data.to_file('Diabetes_data.html')

```




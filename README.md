# Leveraging Machine Learning for Early Prediction and Management of Diabetes -  Big Data Analytics Project 

### Project Overview

Diabetes is a chronic metabolic disorder characterized by elevated blood sugar levels, resulting from insufficient insulin production or ineffective utilization. This disorder is known to be one of the most common chronic diseases affecting people around the globe. 
This research explores the application of machine learning in predicting diabetes, focusing on enhancing early detection and management. The outcomes aim to empower healthcare professionals with actionable insights, contributing to proactive healthcare strategies and improved patient outcomes. 

### Theme: 
Supervised learning Classification Algorithm

### Data Sources:
The Diabetes Dataset of 1000 patients, covers three classes (Diabetic, Non-Diabetic, and Predicted-Diabetic).

[Diabetes Dataset - Mendeley Data](https://data.mendeley.com/datasets/wj9rwkp9c2/1)

The data consists of medical information and laboratory analysis: No. of Patient, Sugar Level Blood, Age, Gender, Creatinine ratio (Cr), Body Mass Index (BMI), Urea, Cholesterol (Chol), Fasting lipid profile, including total, LDL, VLDL, Triglycerides (TG), and HDL Cholesterol, HBA1C, Class (the patient's diabetes disease class may be Diabetic, Non-Diabetic, or Predict-Diabetic).

### Research Questions
- This research focuses on datasets from the laboratory of Medical City Hospital and (the Specializes Center for Endocrinology and Diabetes-Al-Kindy Teaching Hospital)
   - Is there any relationship between the features?
   - How can machine learning algorithms be optimized to accurately predict the onset of diabetes?
   - What feature selection methods enhance the accuracy of diabetes prediction models using machine learning?
     
### Techniques:
- Heatmap/Correlations: To check for relationships/associations between the features.
- Predictive models: Logistic regression, Decision tree, and Naive Bayes
- Chi-square statistics, Correlation Coefficient and Fisher’s score

### Relevant tools:
- Python- for data analysis and visual representation of the datasets.


#### 1- Data Downloading and Inspection
   
Downloading datasets:
[[Diabetes Dataset - Mendeley Data](https://data.mendeley.com/datasets/wj9rwkp9c2/1)](https://data.mendeley.com/datasets/wj9rwkp9c2/1/files/2eb60cac-96b8-46ea-b971-6415e972afc9)


#### 2- Exploratory Data Analysis

1. Install and Import the required packaged into Jupyter notebook.
2. Read and understanding  Dataset_of_Diabetes.csv 
3. View datasets.
4. check data shape i.e. numbers of rows and columns (107, 56).
5. Check datatypes : datatypes were 'datetime', 'int64' and 'object'.
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



 


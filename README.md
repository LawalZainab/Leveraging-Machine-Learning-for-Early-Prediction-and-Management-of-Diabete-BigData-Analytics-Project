# Statistical Analysis on the Rate of Asylum Seekers in Canada between Jan 2015 - Nov 2023 -  Big Data Analytics Project 

### Project Overview

Canada is currently witnessing a surge in the number of people from around the world seeking asylum, which could be due to the unprecedented global displacement. These individuals flee their home countries hoping to obtain refugee status and the right to stay in Canada once they arrive at the claim office. 

Over the years, Canada has gained a reputation for providing a safe haven for refugees, with more than a million people having found refuge there. However, despite Canada's willingness to welcome asylum seekers, it's essential to have proper planning and support systems in place to assist them while their cases are being reviewed.

The purpose of the research is to investigate the patterns, variation, and relationship between the demographic and geographic variables and to predict the number of asylum seekers over time and make data driven recommendations. The results of the research will assist in the planning and resettlement of asylum seekers to avoid the asylum seeker crisis.


### Data Sources
Asylum Claimants- Monthly IRCC Updates- Open Government

Temporary residents in the humanitarian population who request refugee protection upon or after arrival in Canada.
Dataset is between Jan 2015 - Nov 2023.

- 	https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-Asylum-PT_Age.xlsx
- 	https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-Asylum-PT_Gender.xlsx
- 	https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-Asylum-OfficeType_Prov.xlsx
- 	https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-Asylum-Top_25_CITZ_Office_Type_last_month.xlsx
- 	https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-Asylum-Top_25_CITZ_prov_last_month.xlsx

### Tools
- Excel- Data Cleaning/Formatting
- Python - Data Analysis
- Tableau, or Excel or Python for a visual representation of the datasets - To create a report

### Data Cleaning/Preparation

In the initial data preparation phase, the following tasks was performed:
1.  Data Downloading and inspection.
   
 Downloading datasets:
- Asylum-PT_Age;
- Asylum-PT_Gender;
- Asylum-OfficeType_Prov;
- Asylum_Top_25_CITZ_Office_Type_last_month; 
- Asylum-Top_25_CITZ_prov_last_month.

Inspection:
   
- Applying filters and transposing EN_ODP-Asylum-PT_Age dataset OR EN_ODP-Asylum-PT_Gender datasets = provides Total Monthly Asylum Seekers by Territory/Province.
- Applying filters and transposing EN_ODP-Asylum-OfficeType_Prov datasets = provides Total Monthly Asylum Seekers by Claim Office Type.
- Applying filters and transposing EN_ODP-Asylum-Top_25_CITZ_Office_Type_last_month OR Asylum-Top_25_CITZ_prov_last_month = provides Total Monthly Asylum Seekers by Countries of Citizenship.

2. Data cleaning and formatting
   
Asylum Seekers by Claim Office Type (Jan 2015- Nov 2023)
- 	Filtering Asylum-OfficeType_Prov dataset by office claim (Airport, Border, Inland, Other Offices), excluding 'Total' and 'Blanks'.
- 	Transposing the data from row to column in a new sheet.
-  Clean the dataset: delete the Total each year (e.g. 2015 Total), descriptive notes above and below datasets.
-  Format the dataset: add the year to the month to make it consistent.
  

Asylum Seekers by Country of Citizenship (Jan 2015- Nov 2023)
-	Filtering Asylum-Top_25_CITZ_prov_last_month dataset by Country of Citizenship (Mexico, Nigeria, India ….) excluding 'Total’, and 'Blanks'
-	Transposing data from row to column into a new sheet.
-	Clean the dataset:delete the Total each year (e.g., 2015 Total), delete numbering besides each countires, and add to each month.
-	Format the dataset: add the year to the month to make it consistent.
  
Asylum Seekers by Province/Territory (Jan 2015- Nov 2023)
-	Filtering Asylum-PT_Gender dataset by Province/Territory excluding 'Total’, and 'Blanks'.
-	Transposing data from row to column into a new sheet.
-	Clean the dataset: deleting Quarters(column), Total by Quarters(row), Total by Years, and adding the year to the months.
-	Format the dataset: add the year to the month to make it consistent.

### Working Datasets

- Combine the three datasets( Asylum Seekers by Claim Office Type, Asylum Seekers by Country of Citizenship and Asylum Seekers by Province/Territory ) into a single working datasets =  Asylum_Seekers_inCanada_Jan2015-Nov2015.


### Exploratory Data Analysis

1. Install and Import the required packaged into Jupyter notebook.
2. Read Asylum_Seekers_inCanada_Jan2015-Nov2015 dataset into Jupyter notebook. Note: data is in xlsx.
3. View datasets.
4. check data shape i.e. numbers of rows and columns (107, 45).
5. Check datatypes : datatypes were 'datetime', 'int54' and 'object'.
6. Check for missing values : none were detected.
7. Check data summary: Minimum, Maximum, Mean, and the Percentiles (25%, 50%, and 75%) of the datasets.
8. Run ydata profilig on the datasetso

``` Python

pip install ydata-profiling
conda install -c conda-forge ydata-profiling
from ydata_profiling import ProfileReport
import pandas as pd
data = pd.read_excel(r"C:\Users\LawalZa2\Documents\school project\Asylum_Seeker_in Canada_Jan2015-Nov2023.xlsx") # Asylum datasets by Claim Office types
data
data.shape
data.info()
data.isnull().sum()
data.describe()
profile_data = ProfileReport(data)
profile_data

```

### Results
1. Zer0(0) missing cells identified
2. 24 Unsupported variable types were rejected.
3. 20 Numeric variable types used were:
- Highly positively correlated i.e. the variables have a strong relationship with each other.
- Positively Skewed Data Distribution:
  
  Slightly Skewed = Quebec, Iran, China, Other Countries  with Skewness values within the range of  0.5 and 1.
  
  Highly Skewed = Airport, Border, Inland, Ontario, Alberta, Britsh Columbia, Mexico, Nigeria, India, Bangladesh, Pakistan, Haiti, Columbia, Turkey, Sri Lanka, Afganistan with skewness greater than 1

-  Kurtosis
  

![image](https://github.com/LawalZainab/Rate-of-Asylum-Seekers-in-Canada-Big-Data-Analytics-Project/assets/157916270/cff2c31c-131d-4166-b8b0-420586a6e3a8)

![image](https://github.com/LawalZainab/Rate-of-Asylum-Seekers-in-Canada-Big-Data-Analytics-Project/assets/157916270/a781c9ba-bc5f-41e8-8ae7-4665163c3633)

![image](https://github.com/LawalZainab/Rate-of-Asylum-Seekers-in-Canada-Big-Data-Analytics-Project/assets/157916270/69919524-a75e-4f79-b146-0dbd09cb6402)

![image](https://github.com/LawalZainab/Rate-of-Asylum-Seekers-in-Canada-Big-Data-Analytics-Project/assets/157916270/09861377-47dc-4d66-8e82-e314d54f0bf2)

![image](https://github.com/LawalZainab/Rate-of-Asylum-Seekers-in-Canada-Big-Data-Analytics-Project/assets/157916270/e4341467-fea2-4da2-a078-57683cff8e8f)

### From the Result: 24 important variables were excluded( Unsupported)

After investigation, the unsupported variable types were 'Object',  due to the  strings ('--') in the datasets.  This were used to represent values between 0 and 5.
From the Open Government website states 'Please note that in these datasets, the figures have been suppressed or rounded to prevent the identification of individuals when the datasets are compiled and compared with other publicly available statistics. Values between 0 and 5 are shown as “--“ and all other values are rounded to the nearest multiple of 5'

```
import numpy as np
data_new = data.replace('--',np.NaN) ## replacing the sting'--' with NaN'
data_new
data_new.info() # checking datatypes
profile_data_new = ProfileReport(data_new)
profile_data_new

```

### Results from Ydata profile reiterate
1. 427 missing cells
2. 7 Categorical Varaible types: Northwest Territories, Nunavut, Outside Canada, Yukon, Prince Edward Island,  Other Offices, Newfoundland and Labrador; the values on the variables were '0' and NaN
3. 37 Numeric variable types used were:
- Highly positively correlated i.e. the variables have a strong relationship with each other.
- Positively Skewed Data Distribution

Slightly Skewed = Nova Scotia, Quebec, Iran, China, Other Countries  with Skewness values within the range of  0.5 and 1.

Highly Skewed = New Brunswick, Manitoba, Saskatchewan, Kenya, Uganda, Ghana, Rwanda, Romania, Ethiopia, Chad, Burundi, Algeria, Senegal , Syria, Airport, Border, Inland, Ontario, Alberta, Britsh Columbia, Mexico, Nigeria, India, Bangladesh, Pakistan, Haiti, Columbia, Turkey, Sri Lanka, Afganistan with skewness greater than 1.


![image](https://github.com/LawalZainab/Rate-of-Asylum-Seekers-in-Canada-Big-Data-Analytics-Project/assets/157916270/fd7788c2-21b1-4792-85d6-3980382f662f)

![image](https://github.com/LawalZainab/Rate-of-Asylum-Seekers-in-Canada-Big-Data-Analytics-Project/assets/157916270/ae7f2e43-70b3-46fe-978c-62c50bb9087c)

![image](https://github.com/LawalZainab/Rate-of-Asylum-Seekers-in-Canada-Big-Data-Analytics-Project/assets/157916270/debd397b-4b8c-485a-b256-e10159101a05)


### Previous Research
-	There is no previous research where Machine Models were applied to Asylum datasets but there was a Standing Committee on Citizenship and Immigration CIMM - Asylum Trend Data- November 18, 2022, on the Government of Canada website with the following key message.
-Key Messages
-The top three source countries for irregular claimants are Haiti, Turkey, and Colombia.
-In 2022, irregular claimants make up the top method of entry for asylum claimants, whereas, in the past four years, inland claims have represented the highest volumes.
-For the second year in a row, Mexico has been the top source country for all asylum claimants.
-Seasonal increases in asylum claimants occur
-At land borders: July and August
-At airports: May, September and December
-Inland: August to November
-Between the ports: July and August

### Research Questions
- This research focuses on datasets from Jan 2015- Nov 2023
- Check the demographic groups that contribute more to asylum seekers, interaction 
- How do the demographic characteristics vary across different provinces of claim or claim office types? Are there specific demographic groups that are more likely to seek asylum in certain provinces or claim office types?
- How does the demographic group relate to the choice of province or office claim type for lodging an asylum claim? 
- How have the demographics of asylum seekers changed over time, and are there any seasonal patterns in asylum applications? Can we predict future trends in asylum seeker numbers based on historical data? 

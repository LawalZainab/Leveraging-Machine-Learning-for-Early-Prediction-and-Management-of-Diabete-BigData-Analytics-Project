# Statistical Analysis on the Rate of Asylum Seekers in Canada between Jan 2015 - Nov 2023 -  Big Data Analytics Project 

### Project Overview

Canada is currently witnessing a surge in the number of people from around the world seeking asylum, which could be due to the unprecedented global displacement. These individuals flee their home countries hoping to obtain refugee status and the right to stay in Canada once they arrive at the claim office. 

Over the years, Canada has gained a reputation for providing a safe haven for refugees, with more than a million people having found refuge there. However, despite Canada's willingness to welcome asylum seekers, it's essential to have proper planning and support systems in place to assist them while their cases are being reviewed.

The purpose of the research is to investigate the patterns, variation, and relationship between the demographic and geographic variables and to predict the number of asylum seekers over time and make data driven recommendations. The results of the research will assist in the planning and resettlement of asylum seekers to avoid the asylum seeker crisis.


### Data Sources
Asylum Claimants- Monthly IRCC Updates- Open Government

Temporary residents in the humanitarian population who request refugee protection upon or after arrival in Canada.

- 	https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-Asylum-PT_Age.xlsx
- 	https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-Asylum-PT_Gender.xlsx
- 	https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-Asylum-OfficeType_Prov.xlsx
- 	https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-Asylum-Top_25_CITZ_Office_Type_last_month.xlsx
- 	https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-Asylum-Top_25_CITZ_prov_last_month.xlsx

### Tools
- Excel- Data Cleaning 
- Python - Data Analysis
- Tableau, or Excel or Python for a visual representation of the datasets - To create a report

### Data Cleaning/Preparation

In the initial data preparation phase, the following tasks was performed:
1.  Data Downloading and inspection.

Inspection:
   
- Applying filters and transposing EN_ODP-Asylum-PT_Age dataset or EN_ODP-Asylum-PT_Gender datasets provides Total Monthly Asylum Seekers by Territory/Province
- Applying filters and transposing EN_ODP-Asylum-Top_25_CITZ_Office_Type_last_month and Asylum-Top_25_CITZ_prov_last_month = provides Total Monthly Asylum Seekers by country of Citizenship
- Applying filters and transposing EN_ODP-Asylum-OfficeType_Prov datasets provides Total Monthly Asylum Seekers by Claim Office Type


2. Data cleaning and formatting
   
Asylum Seekers by Claim Office Type (Jan 2015- Nov 2023)
- 	Filtering Asylum-OfficeType_Prov dataset by office claim (Airport, Border, Inland, Other Offices), excluding 'Total' and 'Blanks'.
- 	Transposing the data from row to column in a new sheet.
- 	Clean the dataset: delete the Total each year (e.g. 2015 Total) and add the year to each month to make it consistent.
-	The clean dataset provides Asylum Seekers by Claim Office Type.

Asylum Seekers by Country of Citizenship (Jan 2015- Nov 2023)
-	Filtering Asylum-Top_25_CITZ_prov_last_month dataset by Country of Citizenship (Mexico, Nigeria, India ….) excluding 'Total’, and 'Blanks'
-	Transposing data from row to column into a new sheet.
-	Clean the dataset: the Total each year (e.g., 2015 Total), the year added to each month, and the numbering on the country of citizenship deleted.
-	The clean dataset provides Asylum Seekers by Country of Citizenship.

Asylum Seekers by Province/Territory (Jan 2015- Nov 2023)
-	Filtering Asylum-PT_Gender dataset by Province/Territory excluding 'Total’, and 'Blanks'.
-	Transposing data from row to column into a new sheet.
-	Clean the dataset: deleting Quarters(column), Total by Quarters(row), Total by Years, and adding the year to the months.
-	The clean dataset provides Asylum Seekers by Province/Territory.

### Exploratory Data Analysis

- Import dataset into Jupyter notebook.
- View datasets and check number of rows and columns
- Check datatype, we have ‘int64’, ‘object’, and ‘datetime64’
- Note: datatype should be ‘datetime64’ and ‘int64’ when inspected, Columns with ‘object’ datatype contain numbers and strings (“--“).
- Check for missing values: none were detected. 
- Note: The Open Government website states 'Please note that in these datasets, the figures have been suppressed or rounded to prevent the identification of individuals when the datasets are compiled and compared with other publicly available statistics. Values between 0 and 5 are shown as “--“ and all other values are rounded to the nearest multiple of 5'
- “--“is considered missing values.
- Convert data type and Replace missing values
- Replace “--“with NaN, convert the datatype from ‘object’ into ‘floats’ 
- Check for missing values again: columns with missing values and number of missing values identified.
- Replace missing values with the median
- Summary of data provides minimum, maximum, mean, and the percentiles (25%, 50%, and 75%) of the datasets


- Handling Missing values
“--“ was noted to be numbers between 0 and 5, it will be input the median values to replace the msissing values 

``` Python

import pandas as pd
pip install ydata-profiling
conda install -c conda-forge ydata-profiling
from ydata_profiling import ProfileReport
```

- 
-
- 	How do the demographic characteristics vary across different provinces of claim or claim office types? Are there specific demographic groups that are more likely to seek asylum in certain provinces or claim office types?
- 	How does the demographic group relate to the choice of province or office claim type for lodging an asylum claim? 
-  How have the demographics of asylum seekers changed over time, and are there any seasonal patterns in asylum applications? Can we predict future trends in asylum seeker numbers based on historical data? 







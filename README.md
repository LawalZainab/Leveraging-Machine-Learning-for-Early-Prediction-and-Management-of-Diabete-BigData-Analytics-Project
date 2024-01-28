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

#### Asylum-OfficeType_Prov
-  Filter the dataset according to office claim( Airport, Border, Inland, Other Offices), exclude 'Total' and 'Blanks'
-  Create a new sheet
-  Transpose the data from row to column, fill the rest of the years into the new sheet
-  Convert the data from Text to Column by Delimited

#### Asylum-Top_25_CITZ_prov_last_month
- Filter the dataset by Country of Citzenship exclude 'Total', 'Blanks'
- Create a new sheet
- Transpose data from row to column into the new sheet
- Convert the data from Text to Column by Delimited

Combine Prepared Asylum OfficeType-prov data and Asylum Top 25 CITZ-prov-last-month into 'Asylum Claimant by OfficeType, Twenty Five Countries of Citizenship and Claim Year Jan 2015 - Nov 2023'

Convert from excel to CSV file

#### Asylum-PT_Age
- Filter dataset by Province
- Create a new sheet
- Copy total Age Group per year by Province
- Transpose from row to column
- Convert the data from Text to Column by Delimited

#### Asylum-PT_Gender
- Filter dataset by Province
- Create a new sheet
- Copy total Gender per year by Province
- Transpose from row to column
- Convert the data from Text to Column by Delimited

#### Asylum-Top_25_CITZ_Office_Type_last_month
- Filter dataset by Province
- Create a new sheet
- Copy total Countries of Citizenship by Province
- Transpose from row to column
- Convert the data from Text to Column by Delimited

  Combine the three dataset together

  2. Handling Missing data
  3. 

### Exploratory Data Analysis
- 	How do the demographic characteristics vary across different provinces of claim or claim office types? Are there specific demographic groups that are more likely to seek asylum in certain provinces or claim office types?
- 	How does the demographic group relate to the choice of province or office claim type for lodging an asylum claim? 
-   How have the demographics of asylum seekers changed over time, and are there any seasonal patterns in asylum applications? Can we predict future trends in asylum seeker numbers based on historical data? 







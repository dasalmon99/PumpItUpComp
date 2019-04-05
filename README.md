# PumpItUpComp

A RandomForestClassifier for DrivenData.com's Pump it Up: Data Mining the Water Table competition. 

Goal of the challenge is to predict the operating condition of a waterpoint for each record in the dataset.\
Possible conditions are: 
* functional - the waterpoint is operational and there are no repairs needed
* functional needs repair - the waterpoint is operational, but needs repairs
* non functional - the waterpoint is not operational

# General Approach
First step was to drop the many redundant or ill-defined variables (there are, for example, 9 different columns describing geographing location, where a simple lat and long will do). The list of dropped columns:
  * Ill-defined variables or ones containing no information (eg, all entries are the same):'wpt_name', 'num_private', 'recorded_by', 'amount_tsh'
  * Variables that are redundant with other, similarly named columns: 'quantity', 'source_class', 'source_type', 'quality_group', 'scheme_management', 'waterpoint_type', 'extraction_type_class', 'extraction_type', 'payment_type', 'scheme_name'
  * Variables with too many unique categories: 'installer', 'funder'.
  * Variables that describe geographic location: 'district_code', 'subvillage', 'ward', 'region', 'lga', 'region_code'

However, we can use the data from 'lga' and 'region' to handle missing latitude, longitude and gps_height entries (by replacing missing values with the median values of the corresponding 'region', or 'lga', which seems to refer to a subregion). We first replace by median lga, since it the more specific location classifier, then by region for any values that are missed.

![picture alt](https://github.com/dasalmon99/PumpItUpComp/blob/master/lga%20plotted%20by%20location.png "lga plotted by location")
![picture alt](https://github.com/dasalmon99/PumpItUpComp/blob/master/region_code%20plotted%20by%20location.png "region_code plotted by location")

Missing values from the population and construction_year columns were replaced by the medians over the entire dataset.

The final bit of feature engineering came in the form of recasting 'date_recorded' as 'days_elapsed,' which describes the number of days that have elapsed since the entry was recorded to present day. 

Finally, a RandomForestClassifier algorithm was used. A grid search cross validation scheme was used for hyperparameter search, see CrossVal.py. 

# Dataset Variables
amount_tsh - Total static head (amount water available to waterpoint)\
date_recorded - The date the row was entered\
funder - Who funded the well\
gps_height - Altitude of the well\
installer - Organization that installed the well\
longitude - GPS coordinate\
latitude - GPS coordinate\
wpt_name - Name of the waterpoint if there is one\
num_private -\
basin - Geographic water basin\
subvillage - Geographic location\
region - Geographic location\
region_code - Geographic location (coded)\
district_code - Geographic location (coded)\
lga - Geographic location\
ward - Geographic location\
population - Population around the well\
public_meeting - True/False\
recorded_by - Group entering this row of data\
scheme_management - Who operates the waterpoint\
scheme_name - Who operates the waterpoint\
permit - If the waterpoint is permitted\
construction_year - Year the waterpoint was constructed\
extraction_type - The kind of extraction the waterpoint uses\
extraction_type_group - The kind of extraction the waterpoint uses\
extraction_type_class - The kind of extraction the waterpoint uses\
management - How the waterpoint is managed\
management_group - How the waterpoint is managed\
payment - What the water costs\
payment_type - What the water costs\
water_quality - The quality of the water\
quality_group - The quality of the water\
quantity - The quantity of water\
quantity_group - The quantity of water\
source - The source of the water\
source_type - The source of the water\
source_class - The source of the water\
waterpoint_type - The kind of waterpoint\
waterpoint_type_group - The kind of waterpoint

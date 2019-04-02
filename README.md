# PumpItUpComp

A RandomForestClassifier for DrivenData.com's Pump it Up: Data Mining the Water Table competition

Goal of the challenge is to predict the operating condition of a waterpoint for each record in the dataset. Possible conditions are:\\
functional - the waterpoint is operational and there are no repairs needed\
functional needs repair - the waterpoint is operational, but needs repairs\
non functional - the waterpoint is not operational\\

#General Approach
The general approach I had to this data set was first to drop many of the redundant or ill-defined variables(there are, for example, 9 different columns describing geographing location, where a simple lat and long will do. Not much needed to be done in the way of feature engineering. One column is created, 'days_elapsed,' which describes the number of days that have elapsed since the entry was recorded to present day. Finally, many values were missing from the gps_height, latitude, longitude construction year. Missing values were replaced with the medians of those columns. 

The dataset variables are:\
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

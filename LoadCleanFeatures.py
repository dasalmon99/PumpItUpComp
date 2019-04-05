##A RandomForestClassifier for DrivenData.com's Pump it Up: Data Mining the Water Table competition
##Submitted post-deadline for fun/learning, leaderboard place as of 04/04/19: 621/6723 with .8176 accuracy
##Author: Daniel Salmon
##email: dasalmon99@gmail.com


#import functions and libraries
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import LabelEncoder

#Loading Train and Test into pandas DF's
path = ''
TrnLab = pd.read_csv(path + 'TrainLabels.csv')
TrnVal = pd.read_csv(path + 'TrainValues.csv')
Tst = pd.read_csv(path + 'TestVals.csv')

#Combine data for cleaning 0==train, 1==test
Data = pd.concat([TrnVal,Tst],keys=[0,1])

#Dropping repeated or superfluous variables

#Irrelevant variables (or don't know what they mean)
Data = Data.drop(['wpt_name','num_private','recorded_by'],axis=1)
#Data seems to be too sparsely populated (over 70% of entries are 0)
Data = Data.drop('amount_tsh',axis = 1)
#Variables that are redundant with other variables
Data=Data.drop(['quantity','source_class','source_type','quality_group','scheme_management'],axis=1)
Data=Data.drop(['waterpoint_type','extraction_type_class','extraction_type','payment_type','scheme_name'],axis=1)
#Variables with too many unique values/categories
Data=Data.drop(['installer','funder'],axis=1)
#Location data is already included in lat/long
Data=Data.drop(['district_code','subvillage','ward'],axis=1)
Data = Data.drop(['region'],axis = 1)

#Minor feature engineering: recast date_recorded as days elapsed since data taken (days_elapsed)
recorded = pd.to_datetime(Data['date_recorded']) - dt.datetime.now()
Data['days_elapsed'] = recorded.dt.days
Data = Data.drop('date_recorded',axis=1)

#Replace missing values with median for construction_year, population, gps_height, latitude, longitude
Data['population']=Data['population'].replace(0,np.NaN)
med_pop = Data['population'].median()
Data['population']=Data['population'].replace(np.NaN,med_pop)

Data['construction_year']=Data['construction_year'].replace(0,np.NaN)
med_yr = Data['construction_year'].median()
Data['construction_year']=Data['construction_year'].replace(np.NaN,med_yr)

#replacing missing gps data with median values from corresponding
#lga(most specific) then region_code (less specific)

#Replaces any missing values(zeros) with NaN's
Data['longitude']=Data['longitude'].replace(0,np.NaN)
Data['latitude'] = Data['latitude'].replace(-2.000000e-08,np.NaN)
Data.gps_height = Data['gps_height'].replace(0,np.NaN)

#replacing NaN entries with appropriate medians
for reg in Data.lga.unique():
    med_lat = Data.latitude[Data.lga == reg].median()
    med_long = Data.longitude[Data.lga == reg].median()
    med_height = Data.gps_height[Data.lga == reg].median()
    Data.loc[(Data['gps_height'].isnull()) & (Data['lga'] == reg), 'gps_height'] = med_height
    Data.loc[(Data['latitude'].isnull()) & (Data['lga'] == reg), 'latitude'] = med_lat
    Data.loc[(Data['longitude'].isnull()) & (Data['lga'] == reg), 'longitude'] = med_long

for reg in Data.region_code.unique():
    med_lat = Data.latitude[Data.region_code == reg].median()
    med_long = Data.longitude[Data.region_code == reg].median()
    med_height = Data.gps_height[Data.region_code == reg].median()
    Data.loc[(Data['gps_height'].isnull()) & (Data['region_code'] == reg), 'gps_height'] = med_height
    Data.loc[(Data['latitude'].isnull()) & (Data['region_code'] == reg), 'latitude'] = med_lat
    Data.loc[(Data['longitude'].isnull()) & (Data['region_code'] == reg), 'longitude'] = med_long

#There are some regions with no gps_height data, replacing with median height of entire dataset
med_height = Data['gps_height'].median()
Data['gps_height']=Data['gps_height'].replace(np.NaN,med_height)

#Dropping region column now that it has been used for missing data
Data = Data.drop(['lga','region_code'],axis = 1)

#Replace categorical variables with LabelEncoder
#Data['scheme_management'] = Data['scheme_management'].astype(str)
le = LabelEncoder()
sub = Data.dtypes == 'object'
Data.loc[:,sub] = Data.loc[:,sub].apply(le.fit_transform)

#Separate Test and Train, resetting index for consistency
Trn = Data.xs(0)
Tst = Data.xs(1).set_index('id')

#Join sets by id
Trn = Trn.set_index('id').join(TrnLab.set_index('id'))

#Split train predictor values and predicted value
Vals = Trn['status_group']
Trn = Trn.drop('status_group',axis=1)





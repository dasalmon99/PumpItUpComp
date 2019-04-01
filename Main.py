##A RandomForestClassifier for DrivenData.com's Pump it Up: Data Mining the Water Table competition
##Author: Daniel Salmon
##email: dasalmon99@gmail.com

#import data and libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Loading Train and Test into pandas DF's
path = 'C:\\Users\\Daniel\\Documents\\DataSciencePractice\\WaterPumpProblem\\'
TrnLab = pd.read_csv(path + 'TrainLabels.csv')
TrnVal = pd.read_csv(path + 'TrainValues.csv')
Tst = pd.read_csv(path + 'TestVals.csv')

#Combine data for cleaning 0==train, 1==test
Data = pd.concat([TrnVal,Tst],keys=[0,1])

#Dropping repeated or superfluous variables

#Irrelevant variables (or don't know what they mean)
Data = Data.drop(['wpt_name','num_private','recorded_by'],axis=1)
#Variables that are redundant with other variables
Data=Data.drop(['quantity'],axis=1)
Data=Data.drop(['waterpoint_type','extraction_type_class','extraction_type','payment_type','scheme_name'],axis=1)
#Variables with too many unique values/categories
Data=Data.drop(['installer','funder'],axis=1)
#Location data is already included in lat/long
Data=Data.drop(['district_code','region','region_code','subvillage','ward'],axis=1)

#Recast date_recorded as days elapsed since data taken (days_elapsed)
import datetime as dt
recorded = pd.to_datetime(Data['date_recorded']) - dt.datetime.now()
Data['days_elapsed'] = recorded.dt.days
Data = Data.drop('date_recorded',axis=1)

#Replace missing values with mean for construction year and gps height
Data['gps_height']=Data['gps_height'].replace(0,np.NaN)
mean_height = Data['gps_height'].mean()
Data['gps_height']=Data['gps_height'].replace(np.NaN,mean_height)

Data['construction_year']=Data['construction_year'].replace(0,np.NaN)
mean_year = Data['construction_year'].mean()
Data['construction_year']=Data['construction_year'].replace(np.NaN,mean_year)

#Replace categorical variables with LabelEncoder
Data['scheme_management'] = Data['scheme_management'].astype(str)
le = LabelEncoder()
sub = Data.dtypes == 'object'
Data.loc[:,sub] = Data.loc[:,sub].apply(le.fit_transform)

#Separate Test and Train, resetting index for consistency
Trn = Data.xs(0)
Tst = Data.xs(1).set_index('id')

#Join sets by id
Trn = Trn.set_index('id').join(TrnLab.set_index('id'))

#Drop duplicate Rows
Trn = Trn.drop_duplicates()

#Split train predictor values and predicted value
Vals = Trn['status_group']
Trn = Trn.drop('status_group',axis=1)

#Random Forest Classify and predict
clf = RandomForestClassifier(n_estimators=120)
clf.fit(Trn,Vals)
y_pred = clf.predict(Tst)

#Prepare prediction for submission
sub = pd.DataFrame()
Tst = Tst.reset_index()
sub['id'] = Tst['id']
sub['status_group'] = y_pred
sub = sub.set_index('id')

sub.to_csv('Prediction2.csv')

#List feature importance for model
feature_importances = pd.DataFrame(clf.feature_importances_,index = Trn.columns,columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)





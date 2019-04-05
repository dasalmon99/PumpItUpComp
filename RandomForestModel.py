#Random Forest Classify and predict
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=500,max_features='auto',
                                         min_samples_split=8)
clf.fit(Trn,Vals)
y_pred = clf.predict(Tst)

#Prepare prediction for submission
sub = pd.DataFrame()
Tst = Tst.reset_index()
sub['id'] = Tst['id']
sub['status_group'] = y_pred
sub = sub.set_index('id')
sub.to_csv('Prediction.csv')

#List feature importance for model
feature_importances = pd.DataFrame(clf.feature_importances_,index = Trn.columns,columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)

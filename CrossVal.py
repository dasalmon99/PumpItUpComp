#Cross Validation for Hyperparameter Grid Search of RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

for i in ['auto','log2',None]:
    for j in [2,5,8,10]:
        for k in [100,200,300,500]:
        
            clf = RandomForestClassifier(n_estimators=k,max_features=i,min_samples_split=j)
            #clf.fit(Trn,Vals.status_group)
            #y_pred = clf.predict(Tst)
            s = cross_val_score(clf,Trn,scoring=None,
                                cv=5,pre_dispatch='2*n_jobs')
            print(i,j,k,s.mean())

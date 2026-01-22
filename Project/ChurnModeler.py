from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class ChurnModeler:
   
    def __init__(self,radom_state=42):
        self.modeloBase=RandomForestClassifier(class_weight='balanced',random_state=radom_state)
        self.model=None
        
    def train(self,X_train,y_train):
        self.param_grid={'n_estimators':[200,300, 500],'max_depth':[8,10,12,15],
                         'min_samples_split':[2,5,10],
                         'min_samples_leaf':[3,4,5]}
        grid=GridSearchCV(estimator=self.modeloBase,param_grid=self.param_grid,
                          cv=3,n_jobs=-1,scoring='roc_auc', verbose=1)
        grid.fit(X_train,y_train)
        self.model=grid.best_estimator_
    def predict(self,X_test):
        return self.model.predict(X_test)
    
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class ChurnModeler:
    #Hyperparameter Tuning para que el ordenador elija el arbol que mas acertado ha estado
    def __init__(self,radom_state=42):
        self.modeloBase=RandomForestClassifier(class_weight='balanced',random_state=radom_state)
        self.model=None
        
    def train(self,X_train,y_train):
        self.param_grid={'num_estimators':[100,200,300],'max_depth':[10,20,30],
                         'min_samples_split':[2,5,10],
                         'min_samples_leaf':[1,2,4]}
        grid=GridSearchCV(estimator=self.modeloBase,param_grid=self.param_grid,
                          cv=3,n_jobs=-1,scoring='accuracy')
        grid.fit(X_train,y_train)
        self.model=grid.best_estimator_
        
    def predict(self,X_test):
        return self.model.predict(X_test)
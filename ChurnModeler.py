from sklearn.ensemble import RandomForestClassifier

class ChurnModeler:
    def __init__(self, num_estim=100,radom_state=42):
        self.model=RandomForestClassifier(n_estimators=num_estim, class_weight='balanced',random_state=radom_state)
        
    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)
        
    def predict(self,X_test):
        return self.model.predict(X_test)
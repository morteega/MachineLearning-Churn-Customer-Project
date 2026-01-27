import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreProcessor:
    def __init__(self,data):
        self.scaler=StandardScaler()
        self.data=data
        self.label_encoders={}
        self.categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod']
        self.numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
    def encode_categorical(self):
        for col in self.categorical_columns:
            le=LabelEncoder()
            self.data[col]=le.fit_transform(self.data[col])
            self.label_encoders[col]=le
        return self.data
    
    def target_transformation(self):
        self.data['Churn']=self.data['Churn'].map({'Yes':1,'No':0})
        return self.data
    
    def numerical_variables_standraization(self):
        self.fill_null_data()
        self.data[self.numerical_columns]=self.scaler.fit_transform(self.data[self.numerical_columns])
        return self.data
    
    def divide_X_y(self):
        X=self.data.drop(['Churn','customerID'],axis=1)
        y=self.data['Churn']
        return X,y
    
    def scale_numerical(self):
        scaler=StandardScaler()
        columns= ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.data[columns]=scaler.fit_transform(self.data[columns])
        return self.data
    
    def split_data(self, X,y):
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   
    def check_null_data(self):
        return self.data.isnull().sum(),self.data.info()
    
    def fill_null_data(self):
        self.data['TotalCharges']=self.data['TotalCharges'].fillna(0)

    def check_balance(self):
        return self.data['Churn'].value_counts(normalize=True) 
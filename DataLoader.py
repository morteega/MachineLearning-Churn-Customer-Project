import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, path):
        self.file_path=path
        
    def load_data(self):
        try:
            self.data=pd.read_csv(self.file_path)
            self.data['TotalCharges']=pd.to_numeric(self.data['TotalCharges'],errors='coerce')
            return self.data
        except FileNotFoundError:
            print("File wasn`t founf.")
            return None
        except pd.errors.EmptyDataError:
            print("File is empty.")
            return None
            
        
    def cehck_null_data(self):
        return self.data.isnull().sum(),self.data.info()
    
    def fill_null_data(self):
        self.data['TotalCharges']=self.data['TotalCharges'].fillna(0)
        return self.data.isnull().sum()
    # This function fills the null values in totalcharges column with the median of the column but it should be implemente in DataPreprocessor class
    # not here.
    
    def check_balance(self):
        return self.data['Churn'].value_counts(normalize=True)
    # This function is just to know how unbalanced the data is
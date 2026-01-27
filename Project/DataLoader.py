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
            
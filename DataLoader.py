import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.data = None
        
    def load_data(file_path):
        try:
            data=pd.read_csv(file_path)
            return data
        except FileNotFoundError:
            print("File wasnt found.")
            return None
        except pd.errors.EmptyDataError:
            print("File is empty.")
            return None
        
    def process_data(self)
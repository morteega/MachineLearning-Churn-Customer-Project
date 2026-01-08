import pandas as pd
import numpy as np
from DataLoader import DataLoader
from DataPreProcessor import DataPreProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

loader=DataLoader("WA_Fn-UseC_-Telco-Customer-Churn.csv")
data= loader.load_data()
prepocessor=DataPreProcessor(data)
prepocessor.encode_categorical()
prepocessor.target_transformation()
prepocessor.numerical_variables_standraization()
X,y=prepocessor.divide_X_y()
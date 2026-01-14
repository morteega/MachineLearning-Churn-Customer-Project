from DataLoader import DataLoader
from DataPreProcessor import DataPreProcessor
from ChurnModeler import ChurnModeler
from Evaluator import Evaluator

def main():
    print("Welcome to the Churn Prediction Model\n"
          "º----------------------------------º\n")
    data_loader = DataLoader("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data = data_loader.load_data()
    preprocessor = DataPreProcessor()
    preprocessor.data = data
    preprocessor.target_transformation()
    preprocessor.encode_categorical()
    preprocessor.numerical_variables_standraization()
    X, y = preprocessor.divide_X_y()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    modeler = ChurnModeler()
    modeler.train(X_train, y_train)
    predictions = modeler.predict(X_test)
    evaluator = Evaluator(y_test, predictions)
    evaluator.show_metricas()
    evaluator.plot_confusion_matrix()
    
    print("End of the Churn Prediction Model Execution")
    
if __name__ == "__main__":
    main()
from Project.DataLoader import DataLoader
import joblib
from Project.DataPreProcessor import DataPreProcessor
from Project.ChurnModeler import ChurnModeler
from Project.Evaluator import Evaluator
class Churn:
      def __init__(self):
            pass
      
      def run(self):
            print("Welcome to the Churn Prediction Model\n"
                  "º----------------------------------º\n")
            data_loader = DataLoader("WA_Fn-UseC_-Telco-Customer-Churn.csv")
            data = data_loader.load_data()
            preprocessor = DataPreProcessor(data)
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
            evaluator.plot_feature_importance(modeler.model,X.columns)
            
            joblib.dump(modeler.model, 'churn_model.pkl')
            joblib.dump(X.columns.tolist(), 'feature_names.pkl')
    
            print("End of the Churn Prediction Model Execution")
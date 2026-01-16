import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
class Evaluator:
    def __init__(self, y_test, predictions):
        self.y_test = y_test
        self.predictions = predictions
        
    def show_metricas(self):
        print("Prediction results:"+ "\n\n")
        print("General Accuracy: ", accuracy_score(self.y_test, self.predictions)*100, "%"+"\n")
        print("Classification Report: " + "\n")
        print(classification_report(self.y_test, self.predictions))
        
    def plot_confusion_matrix(self):
        cm=confusion_matrix(self.y_test, self.predictions)
        ax=plt.subplots(figsize=(6, 5))
        im=ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        ax.figure.colorbar(im, ax=ax)
        classes=['No Churn', 'Churn']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title='Matriz de Confusión',
               ylabel='Valor Real',
               xlabel='Predicción')
        thresh=cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.show()
       
    def plot_feature_importance(self, model, feature_names):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='skyblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title('¿Qué variables influyen más en el Churn?')
        plt.xlabel('Importancia Relativa')
        plt.show()
        
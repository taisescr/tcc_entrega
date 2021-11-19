import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import numpy as np


def calculaMetricasCV(model,X, y):  
    scores = cross_validate (model, X, y, cv=10, n_jobs=-1,scoring=['accuracy','precision_macro','recall_macro','f1_macro'])
    accuracy = round(scores['test_accuracy'].mean(), 4)*100
    std_acc = round(np.std(scores['test_accuracy']), 4)
    precision = round(scores['test_precision_macro'].mean(), 4)*100
    std_pre = round(np.std(scores['test_precision_macro']), 4)
    recall = round(scores['test_recall_macro'].mean(), 4)*100
    std_rec = round(np.std(scores['test_recall_macro']),4)
    f1 = round(scores['test_f1_macro'].mean(), 4)*100
    std_f1 = round(np.std(scores['test_f1_macro']), 4)
    return accuracy,std_acc, precision,std_pre, recall, std_rec,f1,std_f1

def matrixConfusa(y_test,y_predicted, classes):
    conf_mat = confusion_matrix(y_test, y_predicted, labels=classes)
    matrix = pd.DataFrame(conf_mat,columns=classes)
    matrix.index = classes
    return matrix


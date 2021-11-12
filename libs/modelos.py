
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC  


from libs import metricas
import pickle
import pandas as pd

class classificadores:
    
    random_state=26
    max_iter = 1000
    n_jobs=-1
    LSVC = False 
    MNB = False
    
    
    def __init__(self):

        random_state=26
        max_iter = 10000
        n_jobs=-1
        classes = []
        LSVC = False #bom
        MNB = False
        
        
    def setClasses(self,classes):
        self.classes = classes
        


    def treina(self,model,nomeArquivoModelo,x_train,x_test, y_train, y_test):
        model.fit(x_train,y_train)
        self.salvarModelo(model, nomeArquivoModelo)
        y_predicted = model.predict(x_test)
        return(metricas.matrixConfusa(y_test,y_predicted, self.classes))

    def salvarModelo(self,model,nomeArquivo):
        pickle.dump(model, open(nomeArquivo, 'wb'))
 
    def carregarModelo(self,nomeArquivo):
        modelo = pickle.load(open(nomeArquivo, 'rb'))
        return(modelo)

    def classificar(self,x_total,y_total):
        
        resultados = []
       
        if self.LSVC:
            LSVC = LinearSVC(random_state=self.random_state, max_iter=1000)
            accuracy, std_acc, precision,std_pre, recall,std_rec, f1, std_f1 = metricas.calculaMetricasCV(LSVC,x_total, y_total)
            resultados.append(['LinearSVC',accuracy,std_acc, precision,std_pre, recall,std_rec, f1,std_f1])
       
        if self.MNB:
            MNB = MultinomialNB()
            accuracy, std_acc, precision,std_pre, recall,std_rec, f1, std_f1 = metricas.calculaMetricasCV(MNB,x_total, y_total)
            resultados.append(['MultinomialNB',accuracy,std_acc, precision,std_pre, recall,std_rec, f1,std_f1])

        return (resultados)

    
    def best(self, nomemodelo, x_train,x_test,y_train, y_test):        
        if (nomemodelo=='LinearSVC'):
            modelo = LinearSVC(random_state=self.random_state, max_iter=10000)
        if (nomemodelo=='MultinomialNB'):
            modelo = MultinomialNB()
        matrix = self.treina(modelo,nomemodelo,x_train,x_test, y_train, y_test)
        return matrix
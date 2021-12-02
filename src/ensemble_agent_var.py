# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 22:03:08 2021

@author: MyPc
"""
from operative_agent import operative_agent_model
from sklearn.ensemble import RandomForestClassifier
import operative_agent
import data_manipulation
import numpy as np
from sklearn.metrics import confusion_matrix
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
import forest_variation
from sklearn import metrics
from sklearn.metrics import brier_score_loss



class ensemble_agent_model:
    def __init__(self, epoch=None, dataset=None, model=None,  model_list=None, rul_model_list=None, opt_function=None):
         if epoch is None:
             self.epoch=2
         else:
             self.epoch=epoch

            
         if model_list is None:
             self.model_list=[operative_agent.operative_agent_model('logistic', role='std'), operative_agent.operative_agent_model('svm', role='std')]
             self.rul_model_list=[operative_agent.operative_agent_model('ridge', role='magnet'), operative_agent.operative_agent_model('svm', role='magnet')]
         else:
             for elem in model_list:
                 self.model_list.append(operative_agent_model(elem, role='std'))
             for elem in rul_model_list:
                 self.rul_model_list.append(operative_agent_model(elem, role='magnet'))
         if dataset is None:
            self.dataset=data_manipulation.Data_ensemble(self.model_list, self.rul_model_list)
         else:
            self.dataset=dataset    
         if model is None:
             self.model = RandomForestClassifier(random_state=0)
         else:
             self.model=model
  
         self.confusion_matrix=0 #matrice di confusione dell'ultima previsione
         self.f1_pred=0 #f1 ultima previsione
         self.last_pred=[]
         self.last_pred_proba=[]
         self.prev_pred=[]
         self.prev_pred_proba=[]
         self.scaler=0
         self.n_estimators=100
         self.estimator_list=[100, 200, 300, 400]
         self.criterion="gini"
         self.criterion_list=["gini", "entropy"]
         self.min_sample_split=2
         self.n=self.dataset.data.machine_number
         self.confidence=0
         self.cl_error=0
         self.savings=0
         self.savings_pred=0
         self.sigma=1
         self.f1=0
         self.precision=0
         self.recall=0
         self.accuracy=0
         self.corr_coef=0
         self.brier=0
         self.pr_rec=[]
         self.perc_effort=100
         self.ROC=[]
    def give_prediction(self):#QUI SI DEVE FARE UN CALCOLO DELLA PREVISIONE DI RISPARMIO
        return 0
    def give_feedback(self):
        y_test = self.dataset.Y_train
        self.savings+=forest_variation.save_calculation(self.last_pred, y_test)
        self.savings_pred=forest_variation.save_calculation(self.last_pred, y_test)
        return  np.random.normal(self.saving_pred, self.sigma, 1) 
        


    
    def recall_calculation(self, matr):
        tp=0
        fn=0
        fn += matr[0][1]
        tp += matr[0][0]
        return (tp/(tp+fn))
    
    def precision_calculation(self, matr):
        tp=0
        fp=0
        fp += matr[1][0]
        tp += matr[0][0]
        return (tp/(tp+fp))
    
    def accuracy_calculation(self, matr):
        tp=0
        tn=0
        nn=0
        pp=0
        tn += matr[1][1]
        tp += matr[0][0]
        nn += (matr[1][1]+matr[1][0])
        pp += (matr[0][0]+matr[0][1])
            
        TPR=tp/pp
        TNR=tn/nn
        return (TPR+TNR/2)
    
    def corr_coef_calculation(self, matr):
        tp=0
        fn=0
        fp=0
        tn=0
        tn += matr[1][1]
        tp += matr[0][0]
        fn += matr[0][1]
        fp += matr[1][0]
            
        num = (tp*tn-fp*fn)
        den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return (num/den)
    
    def roc_calculation (self, y, score, pos_label=2):
        roc_list=[]
        fpr, tpr, thresholds = metrics.roc_curve(y, score, pos_label)
        lista=[]
        lista.append(fpr)
        lista.append(tpr)
        lista.append(thresholds)
        roc_list.append(lista)
        return roc_list
        
    def pr_rec_calculation (self, y, score):
        prc_list=[]
        pr, rec, thr = metrics.roc_curve(y, score)
        lista=[]
        lista.append(pr)
        lista.append(rec)
        lista.append(thr)
        prc_list.append(lista)
        return prc_list
    
    def brier(self, y, score):
       
       
        brier=brier_score_loss(y, score)
        return brier
    
    def perc_effort_calculation(self, matr, multiplicative=False):
         if multiplicative==False:
             multiplicative=1
         else:
             multiplicative=100
         real_effort=0
         actual_effort=0
         tn=0
         tp=0
         fn=0
         fp=0
         tn += matr[1][1]
         tp += matr[0][0]
         fn += matr[0][1]
         fp += matr[1][0]
            
         real_effort=tn+tp+fn+fp
         actual_effort=tp+fp
         
         return(((real_effort-actual_effort)/real_effort)*multiplicative)
    

    def calculate_metrics(self, matrix_list, y=None, score=None):

        self.f1=self.f1_calculation(matrix_list)
        self.precision=self.precision_calculation(matrix_list)
        self.recall=self.recall_calculation(matrix_list)
        self.accuracy=self.accuracy_calculation(matrix_list)
        self.corr_coef=self.corr_coef_calculation(matrix_list)
        self.perc_effort=self.perc_effort_calculation(matrix_list)
        self.ROC=self.roc_calculation(y, score)
        self.pr_rec=self.pr_rec_calculation(y, score)
        self.brier=self.brier_calculation(y, score)
        
        
    def feature_selection(self):
        max_save=0
        feature_selected=[]
        if(len(list(self.dataset.X_train.columns))<=4):
            for features_subset in data_manipulation.data.features.subsets(list(self.dataset.X_train_prev.columns)):
                save_subset=0
                #estraggo il frame per il train          
                x_train = self.dataset.X_train_prev[features_subset]
                y_train = self.dataset.Y_train_prev[features_subset]
                #estraggo i frame per il test
                x_test = self.dataset.X_train[features_subset]
                y_test = self.dataset.Y_train[features_subset]
                #preparazione modello
                clf = RandomForestClassifier(random_state=0)
                #esegue train e classificazione
                clf.fit(x_train, y_train)
                y_pred=clf.predict(x_test) 
                save_subset=forest_variation.save_calculation(y_pred, y_test)
                if(save_subset>max_save):
                    max_save=save_subset
                    feature_selected=features_subset
        else: #approccio bottom up per troppi insiemi
            feature_subset=[]
            for elem in list(self.dataset.X_train.columns):
                feature_subset.append(elem)
                #estraggo il frame per il train          
                x_train = self.dataset.X_train_prev[feature_subset]
                y_train = self.dataset.Y_train_prev
                #estraggo i frame per il test
                x_test = self.dataset.X_train[feature_subset]
                y_test = self.dataset.Y_train
                #preparazione modello
                clf = RandomForestClassifier(random_state=0)
                #esegue train e classificazione
                clf.fit(x_train, y_train)
                y_pred=clf.predict(x_test) 
                save_subset=forest_variation.save_calculation(y_pred, y_test)
                if(save_subset>max_save):
                    max_save=save_subset
                    feature_selected=features_subset
                    
        if not feature_selected:
            feature_selected.append(list(self.dataset.X_train.columns)[1])


        self.dataset.features.selected_features=feature_selected
        self.min_sample_split=int(math.sqrt(len(feature_selected)))
        
  
    
    
    def parameter_selection(self):
             max_save=0
             for parameter in self.estimator_list:
                #estraggo il frame per il train          
                x_train = self.dataset.X_train_prev[self.dataset.features.selected_features]
                self.scaler.fit_transform(x_train)
                y_train = self.dataset.Y_train_prev
                #estraggo i frame per il test
                x_test = self.dataset.X_train[self.dataset.features.selected_features]
                self.scaler.fit_transform(x_test)
                y_test = self.dataset.Y_train
                #preparazione modello
                clf = RandomForestClassifier(n_estimators=parameter, random_state=0)
                #esegue train e classificazione
                clf.fit(x_train, y_train)
                y_pred=clf.predict(x_test) 
                save_parameter=forest_variation.save_calculation(y_pred, y_test)
                if(save_parameter>max_save):
                    max_save=save_parameter
                    self.n_estimator=parameter

                    
             max_save=0
             for parameter in self.criterion_list:
                #estraggo il frame per il train          
                x_train = self.dataset.X_train_prev[self.dataset.features.selected_features]
                self.scaler.fit_transform(x_train)
                y_train = self.dataset.Y_train_prev
                #estraggo i frame per il test
                x_test = self.dataset.X_train[self.dataset.features.selected_features]
                self.scaler.fit_transform(x_test)
                y_test = self.dataset.Y_train
                #preparazione modello
                clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=parameter, random_state=0)
                #esegue train e classificazione
                clf.fit(x_train, y_train)
                y_pred=clf.predict(x_test) 
                save_parameter=forest_variation.save_calculation(y_pred, y_test)
                if(save_parameter>max_save):
                    max_save=save_parameter
                    self.criterion=parameter
                    self.confusion_matrix=confusion_matrix(y_test, y_pred)
            
             return 0
    
    def scaler_selection(self):
         scalers=[StandardScaler(), MinMaxScaler(), MaxAbsScaler()]
         max_save=0
         for scaler in scalers:
            #estraggo il frame per il train          
            x_train = self.dataset.X_train_prev[self.dataset.features.selected_features]
            scaler.fit_transform(x_train)
            y_train = self.dataset.Y_train_prev
            #estraggo i frame per il test
            x_test = self.dataset.X_train[self.dataset.features.selected_features]
            scaler.fit_transform(x_test)
            y_test = self.dataset.Y_train
            #preparazione modello
            clf = RandomForestClassifier(random_state=0)
            #esegue train e classificazione
            clf.fit(x_train, y_train)
            y_pred=clf.predict(x_test) 
            save_scaler=forest_variation.save_calculation(y_pred, y_test)
            if(save_scaler>max_save):
                max_save=save_scaler
                self.scaler_selected=scaler
         return 0
    
    def performance_calculation(self):
           #estraggo il frame per il train          
           x_train = self.dataset.X_train_prev[self.dataset.features.selected_features]
           self.scaler.fit_transform(x_train)
           y_train = self.dataset.Y_train_prev
           #estraggo i frame per il test
           x_test = self.dataset.X_train[self.dataset.features.selected_features]
           self.scaler.fit_transform(x_test)
           y_test = self.dataset.Y_train
           #preparazione modello
           clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, random_state=0)
           #esegue train e classificazione
           clf.fit(x_train, y_train)
           y_pred=clf.predict(x_test) 
           y=y_test
           score=clf.predict_proba(x_test)
           matr=confusion_matrix(y, y_pred, labels=[1, 0])
           self.cl_error=self.cl_error(y_pred, y_test)
           self.confidence=self.calculate_confidence(self.cl_error)
           self.calculate_metrics(matr, y, score)
       
    def cl_error(self, y_pred, y):
        numerator=0
        total=0
        matr=confusion_matrix(y, y_pred, labels=[1, 0])
        total += matr[1][0]+matr[0][1]+matr[1][1]+matr[0][0]
        numerator += matr[1][1] + matr[0][0]
        return (1-((numerator/total)))
    
    #VALORE T CON 26 MACCHINE DA MODIFICARE
    def calculate_confidence(self, error,  t=1.708, n=26):
         return t * math.sqrt( (error * (1 - error)) / n)
    
    def train_model(self):
         if(self.epoch>2):
             self.feature_selection()
             self.scaler_selection()
             self.parameter_selection()
             self.performance_calculation()
             
         if(self.epoch>2):
             self.scaler.fit_transform(self.dataset.X_train)
             self.model=RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, random_state=0)
         self.model.fit(self.dataset.X_train[self.dataset.features.selected_features], self.dataset.Y_train)
         
    def test_model(self):
       if(self.epoch>2):
             self.scaler.fit_transform(self.dataset.X_test)
       self.last_pred.append(self.model.predict(self.dataset.X_test[self.dataset.features.selected_features]))
       self.last_pred_proba.append(self.model.predict_proba(self.dataset.X_test[self.dataset.features.selected_features]))
    
    def operative_cycle(self):
        for model in self.model_list:
            model.operative_cycle()
        self.dataset.obtain_data(self.epoch, train=True)
        
        self.train_model()
        self.dataset.obtain_data(self.epoch, train=False)
        self.test_model()
        self.give_prediction()
        self.epoch+=1
        self.model = RandomForestClassifier(random_state=0)

ensemble=ensemble_agent_model()
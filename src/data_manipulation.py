# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 22:04:05 2021

@author: MyPc
"""


#LIBRARIES
import math
import statistics
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
import json
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
import itertools




#CLASSES
#list dict
class ListDict(object):
    def __init__(self):
        self.position_to_item = {}
        self.counter = 0

    def add(self, item):
        self.position_to_item[self.counter] = item
        self.counter+=1

    def get(self):
        choice=np.random.choice(range(self.counter))
        return self.position_to_item[choice]
    
#features
class Features:

  def __init__(self, role=None, targets=None, selected=None):
    self.feature_requested={
        "mean":False,
        "rms":False,
        "var":False,
        "skew":False,
        "curt":False,
        "maxv":False,
        "mad":False,
        "quant":False,
        "iqr":False,
        "peak":False,
        "trim":False,
        "cf":False
    }
    self.feature_subject={
        "mean":[],
         "rms":[],
        "var":[],
        "skew":[],
        "curt":[],
        "maxv":[],
        "mad":[],
        "quant":[],
        "iqr":[],
        "peak":[],
        "trim":[],
        "cf":[]
    }
    self.feature_functions={
        "mean":self.mean_calculation,
        "rms":self.rms,
        "var":self.var,
        "skew":self.skew,
        "curt":self.curt,
        "maxv":self.maxv,
        "mad":self.mad,
        "quant":self.quant,
        "iqr":self.iqr,
        "peak":self.peak,
        "trim":self.trim,
        "cf":self.cf
        
    }
    

    if targets is None:
        if role==None:
            self.targets=[ "bearings", "electricity_1", "electricity_2"]
        elif role=="magnet":
            self.targets=["magnet"]
        elif role=='general':
            self.targets=[ "bearings", "electricity_1", "electricity_2", "magnet"]
    else:
        self.targets=targets 
        
    if selected is not None:
        self.selected_features=selected
        
    self.extracted_features=[]#extracted features from all features
    self.total_features=[] #all features
    self.selected_features=[] #selected features from all features


  #subsets
  def subsets(self, features=None):
    features_combo=[[]]
    if features is None:
        feature_set=set(self.total_features)
    else:
        feature_set=set(features)
    length=len(feature_set)
    #ritorna i sottoinsiemi
    for i in range(1, length):
        subset = itertools.combinations(feature_set, i)#ottiene tutti i subset di lunghezza i (https://www.kite.com/python/answers/how-to-find-all-subsets-with-length-n-of-a-given-set-in-python)
        for elem in subset:
            features_combo.append(list(elem))
    return features_combo   

  #metodi di calcolo features
  #mena
  def mean_calculation(self, df, machine, value, day): #df è il dataframe, value è  su quale dei dati raccolti  (quale colonna) vogliamo calcolare la feature media, day è il giorno per cui calcoliamo la feature, machine è la macchina per quale calcoliamo la feature
    df1=df[(df.machine==machine)&(df.day<=day)]
    return df1[value].mean()
  #root mean square
  def rms(self, df, machine, value, day):
    df1=df[(df.machine==machine)&(df.day<=day)]
    return math.sqrt(df1[value].mean())
  #variance
  def var(self, df, machine, value, day):
    df1=df[(df.machine==machine)&(df.day<=day)]
    return df1[value].var()
  #skewness
  def skew(self, df, machine, value, day):
    df1=df[(df.machine==machine)&(df.day<=day)]
    return df1[value].skew()
  #kurtosis
  def curt(self, df, machine, value, day):
    df1=df[(df.machine==machine)&(df.day<=day)]
    return stats.kurtosis(df1[value])
  #max value
  def maxv(self, df, machine, value, day):
    df1=df[(df.machine==machine)&(df.day<=day)]
    return df1[value].max()
  #mean absoulte deviation
  def mad(self, df, machine, value, day):
    df1=df[(df.machine==machine)&(df.day<=day)]
    return df1[value].mad()
  #qualite
  def quant(self, df, machine, value, day, q_value=0.5):
    df1=df[(df.machine==machine)&(df.day<=day)]
    return df1[value].quantile(q_value)
  #interquarile range
  def iqr(self, df, machine, value, day, q_value1=0.25, q_value2=0.75):
    df1=df[(df.machine==machine)&(df.day<=day)]
    return df1[value].quantile(q_value2)-df1[value].quantile(q_value1)
  #peak value
  def peak(self, df, machine, value, day):
    df1=df[(df.machine==machine)&(df.day<=day)]
    df1=df1.abs()
    return df1[value].max()
  #trimmed mean
  def trim(self, df, machine, value, day, trim_value=0.05):
    df1=df[(df.machine==machine)&(df.day<=day)]
    return stats.trim_mean(df1[value], trim_value)
  #crest factor
  def cf(self, df, machine, value, day):
    df1=df[(df.machine==machine)&(df.day<=day)]
    rms=math.sqrt(df1[value].mean())
    df1=df1.abs()
    cf=df1[value].max()/rms
    return cf
  #set calculated features
  def set_features(self, key=None, values=None, function=None):
      if values is None:
          self.feature_requested={
               "rms":True,
               "var":True,
               "skew":True,
               "curt":True,
               "maxv":True,
               "mad":True,
               "quant":True,
               "iqr":True,
               "peak":True,
               "trim":True,
               "cf":False
          }
      else:
          if function is None:
              self.feature_requested[key]=values
          else:
              self.feature_function[key]=function
  #set calculated feature subject
  def set_feature_subject(self, key, values):
      self.feature_subject[key]=values

#data for operative agent class
class Data_operative:
    def __init__(self, outlier=1, balancing=True, role=None, df=None, features=None):
      #obtain outlier and balancing
      self.outlier=outlier
      self.balancing=balancing
      if role is None:#set role of classifier that use that data
          self.role="std"
      else:
          self.role=role
      if df is None:#set dataframe to stanard if none
          self.df=self.use_standard_dataframe()
      else:
          self.df=df
      if features is None:#set features as satandard if none
          if ((self.role=="std")):
              self.features=Features()
          else:
              self.features=Features(role)
          self.features.set_features()
          #clustering feature
          
          self.clustering_feature(self.df)
          
          for element in self.df.columns.values.tolist(): #update total features
              if((element != "day") or(element !="machine")or(element != "electricity_1")or(element != "electricity_2")or(element != "magnet")or(element != "bearings")):
                  self.features.total_features.append(element)
                  
          #add subject to feature calculation
          self.features.set_feature_subject("mean", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel',  'temperature_external', 'current'])
          self.features.set_feature_subject("var", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel', 'temperature_external', 'current'])
          self.features.set_feature_subject("skew", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel',  'temperature_external', 'current'])
          self.features.set_feature_subject("curt", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel',  'temperature_external', 'current']) 
          self.features.set_feature_subject("maxv", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel',  'temperature_external', 'current'])
          self.features.set_feature_subject("mad", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel',  'temperature_external', 'current'])     
          self.features.set_feature_subject("quant", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel',  'temperature_external', 'current'])
          self.features.set_feature_subject("iqr", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel',  'temperature_external', 'current'])
          self.features.set_feature_subject("peak", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel', 'temperature_external', 'current'])
          self.features.set_feature_subject("trim", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel',  'temperature_external', 'current'])     
          self.features.set_feature_subject("cf", ['s_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 'sonic_rmslog', 'vib_x_acc', 'vib_x_kurt', 'vib_x_peak', 'vib_x_vel', 'vib_y_acc', 'vib_y_kurt', 'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak', 'vib_z_vel',  'temperature_external', 'current'])
          #it s very time consuming so i removed it for testing
         #self.feature_extraction()
          
      else:
          self.features=features
      if(self.role=='magnet'):
            self.features.targets_binarized=['0', '0.5', '0.7', '1.0']
            self.df.loc[self.df['magnet']==0,'0'] = 1
            self.df['0'].fillna(0, inplace=True)
            self.df.loc[self.df['magnet']==0.5,'0.5'] = 1
            self.df['0.5'].fillna(0, inplace=True)
            self.df.loc[self.df['magnet']==1,'1.0'] = 1
            self.df['1.0'].fillna(0, inplace=True)
            self.df.loc[self.df['magnet']==0.7,'0.7'] = 1
            self.df['0.7'].fillna(0, inplace=True)
            self.features.targets_nb=['magnet_nb']
            self.df.loc[self.df['magnet']==0,'magnet_nb'] = '0'
            self.df.loc[self.df['magnet']==0.5,'magnet_nb'] = '0.5'
            self.df.loc[self.df['magnet']==1,'magnet_nb'] = '1.0'
            self.df.loc[self.df['magnet']==0.7,'magnet_nb'] = '0.7'
            self.df['magnet_nb'].fillna('n/a', inplace=True)
            
      #obtain machine number
      self.machine_number=self.df['machine'].nunique()
      
     
    #recover data from asystom platform . DA FINIRE??? 
    def retrieve_data(self):  
        return 0

    #generate AGWN noise
    def noise_generation(self, df):
        
       df_base = df.copy()
       df = df[df.label.isin(['bearings-1', 'bearings-2', 'magnet-partial', 'magnet-1'])]
       df = df.drop("label", axis=1)
       for col in df.columns:
          if col != 'time':
              std = statistics.stdev(df[col])
              noise = np.random.normal(0, std, len(df))
              df[col] += noise
       df["label"] = df_base.label
       return df



    #remove outlier
    def remove_outlier(self, df, outlier=1):
        if(outlier==1): #first way use 3 sigma
            for label in df.label.unique():
                dff = df[df.label == label]
                # dff = dff.reset_index()
                dff = dff.drop(["label", "vib_z_f1", "vib_z_f2", "vib_z_f3"], axis=1)
                dff = dff.filter([
                    's_00', 's_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08',
                    's_09', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17',
                    's_18', 's_19',
                    'sonic_custom', 'sonic_rmslog', 'vib_x_acc',
                    #'vib_x_f1', 'vib_x_f2', 'vib_x_f3',
                    'vib_x_kurt', 'vib_x_peak', 'vib_x_vel',
                    'vib_y_acc', 'vib_y_f1', 'vib_y_f2', 'vib_y_f3', 'vib_y_kurt',
                    'vib_y_peak', 'vib_y_vel', 'vib_z_acc', 'vib_z_kurt', 'vib_z_peak',
                    'vib_z_vel', 'temperature_external', 'current'
                ])
                dff = dff[~(np.abs(stats.zscore(dff)) < 3).all(axis=1)]
                df = df.drop(dff.index)
        else: #second way using intequantile range
                dff = df.drop([ "label", "vib_z_f1", "vib_z_f2", "vib_z_f3"], axis=1)
                Q1 = dff.quantile(0.25)
                Q3 = dff.quantile(0.75)
                IQR = Q3 - Q1
                dff= dff[~((dff < (Q1 - 1.5 * IQR)) |(dff > (Q3 + 1.5 * IQR))).any(axis=1)]
                df = pd.merge(df, dff, how='inner')

    #visualization methods
   #strip graph creation
    def strip_graph(self, df):
        c=[sns.color_palette()[x] for x in pd.factorize(df.label)[0]]
        # for col in df.columns[1:-1]:
        for col in 's_11', 's_17', 'vib_x_kurt', 'current': # df.columns[1:-1]
            fig, (ax1, ax2) = plt.subplots(1,2)
            sns.stripplot(x="label", y=col, data=df, jitter=0.2, c=c, order=df.label.unique(), ax=ax1)
            sns.stripplot(x="label", y=col, data=df, jitter=0.2, c=c, order=df.label.unique(), ax=ax2)
            fig.set_size_inches(20, 6)
            plt.title(col)
            plt.show()
    #visualize data as box plot
    def visualize(self, df):
        for col in df.columns[1:-1]:
            fig, (ax1, ax2) = plt.subplots(1,2)
            for label in df.label.unique():
                dff = df[df.label == label][col].reset_index()
                ax2.plot(dff[col], label=label)
                dff = df[df.label == label][col].reset_index()
                ax1.plot(dff[col], label=label)
            fig.set_size_inches(16, 6)
            plt.title(col)
            plt.show()
    #PCA calculation
    def PCA(self, df):
        df_dr = df.drop(["label"], axis=1)
        reducer = PCA()
        embedding= reducer.fit_transform(df_dr)
        c=[sns.color_palette()[x] for x in pd.factorize(df.label)[0]]
        df_dr = df.drop(["label"], axis=1)
        reducer = PCA()
        embedding_base= reducer.fit_transform(df_dr)
        c_base=[sns.color_palette()[x] for x in pd.factorize(df.label)[0]]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1.scatter(embedding_base[:, 0], embedding_base[:, 1], c=c_base)
        ax2.scatter(embedding[:, 0], embedding[:, 1], c=c)
        plt.show()
    
    #data generation
    def data_generation(self, df):
        X = df.drop(["label"], axis=1)
        y = df.label
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        df = X_resampled
        df["label"] = y_resampled
        
    def clustering_feature(self, df, n_clusters=4, visualize_elbow=False, visualize_clusters=False):
            
            numeric_df = df.select_dtypes('number')
            numeric_df =numeric_df.drop('machine', 1)
            numeric_df =numeric_df.drop('day', 1)
            if(self.role=='std'):
                numeric_df =numeric_df.drop('bearings', 1)
                numeric_df =numeric_df.drop('electricity_1', 1)
                numeric_df =numeric_df.drop('electricity_2', 1)
            elif(self.role=='magnet'):
                numeric_df =numeric_df.drop('magnet', 1)
            elif(self.role=='general'):
                numeric_df =numeric_df.drop('bearings', 1)
                numeric_df =numeric_df.drop('electricity_1', 1)
                numeric_df =numeric_df.drop('electricity_2', 1)
                numeric_df =numeric_df.drop('magnet', 1)
            
            if (visualize_elbow==True):#visualize cluster to check elbow method functioning
                #https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
                #distortion = It is calculated as the average of the squared distances from the cluster centers of the respective clusters. Typically, the Euclidean distance metric is used.
                #inertia = It is the sum of squared distances of samples to their closest cluster center.
                distortions = []
                inertias = []
                mapping1 = {}
                mapping2 = {}
                K = range(1,10)
                #testing for all possible clusters
                for k in tqdm(K):
                    #building and fitting model
                    kmean_model = KMeans(init='k-means++',n_clusters=k)
                    kmean_model.fit(numeric_df)
                    
                    distortions.append(sum(np.min(cdist(numeric_df, kmean_model.cluster_centers_,
                                                        'euclidean'), axis=1)) / numeric_df.shape[0])
                    inertias.append(kmean_model.inertia_)
                    
                    mapping1[k] = sum(np.min(cdist(numeric_df, kmean_model.cluster_centers_,
                                                   'euclidean'), axis=1)) / numeric_df.shape[0]
                    mapping2[k] = kmean_model.inertia_
                
                for key, val in mapping1.items(): print(f'{key} : {val}')
                plt.plot(K, distortions, 'bx-')
                plt.xlabel('Values of K')
                plt.ylabel('Distortion')
                plt.title('The Elbow Method using Distortion')
                plt.show()
                for key, val in mapping2.items():
                    print(f'{key} : {val}')
                plt.plot(K, inertias, 'bx-')
                plt.xlabel('Values of K')
                plt.ylabel('Inertia')
                plt.title('The Elbow Method using Inertia')
                plt.show()
                
            kmeans_0 = KMeans(init='k-means++',n_clusters=4)
            kmeans_0.fit(numeric_df)#???
            set(kmeans_0.labels_)
            
            if (visualize_clusters==True):
                sc = StandardScaler()
                X_norm = sc.fit_transform(numeric_df)
                #pca for visualization in 3 dimensions
                pca = PCA(n_components=3)
                components = pca.fit_transform(X_norm)
                #explained variance of the 3 components (how much well we're able to see)
                print(pca.explained_variance_ratio_)
                fig = px.scatter_3d(components, 
                            x=0,y=1,z=2, 
                            color=kmeans_0.labels_,
                            labels = {'0':'PC1','1':'PC2','2':'PC3'}
                           )
                fig.show()
    
            df['kmeans_label'] = kmeans_0.labels_
            self.features.total_features.append('kmeans_label')

    
    #extract all features
    def feature_extraction(self, features=None):
        if features is None:
            features=self.features
        for key, value in features.feature_requested.items():
            if (value==True):
                for subject in features.feature_subject[key]:
                    self.df[key + '' + subject]=self.df.apply(lambda x:(features.feature_functions[key](self.df, x['machine'], subject, x['day'])), axis=1)
                    self.features.extracted_features.append(key + '' + subject)
                    self.features.total_features.append(key + '' + subject)
        return self.df  
    
    
        
    #create standard dataframe for operative agent
    def use_standard_dataframe(self):
        
        #retrieving data from asystom platform and save as jason file
        self.retrieve_data()
        #opening of data saved
        with open("28-09_19-11_alldata.json", "r") as f:
            data = json.load(f)
        #creation of dataframe from file
        pd.DataFrame(data)
        dfs = []
        for result in data["results"]:
            serie = result["series"][0]
            for serie in result["series"]:
                df = pd.DataFrame(serie["values"], columns=serie["columns"])
                df.name = serie["name"]
                dfs.append(df)
                
        df = dfs[0]
        df_temp = dfs[2]
        df_curr = dfs[3]
        df = df.drop([
            'client', "GW", "device", "mileage", "vibra_custom",
            'vib_x_root', 'vib_y_root', 'vib_z_root',
        ], axis=1)
        df_temp = df_temp.drop(['client', "GW", "device", 'weekday'], axis=1)
        df_curr = df_curr.drop(['client', "GW", "device", 'weekday'], axis=1)
        #join dataframe
        df = df.set_index('time')
        df = df.rename(columns={'temp': 'temperature_surface'})
        df_temp = df_temp.set_index('time').rename(columns={"value": "temperature_external"})
        df_curr = df_curr.set_index('time').rename(columns={"value": "current"})
        df = df.join(df_temp)
        df = df.join(df_curr)
        df = df.reset_index()
        df["time"] = pd.to_datetime(df["time"], unit='ms')
        df.loc[(df["time"] >= "2021-10-09 00:00") & (df["time"] <= "2021-10-12 00:00"), "label"] = "working-engine-1"
        df.loc[(df["time"] >= "2021-10-27 17:30") & (df["time"] <= "2021-10-29 00:00"), "label"] = "bearings-1"
        df.loc[(df["time"] >= "2021-11-03 16:00") & (df["time"] <= "2021-11-05 11:00"), "label"] = "working-engine-2" 
        df.loc[(df["time"] >= "2021-11-05 13:30") & (df["time"] <= "2021-11-05 23:59"), "label"] = "bearings-2"
        # 5th to 10th nothing happened
        df.loc[(df["time"] >= "2021-11-10 16:30") & (df["time"] <= "2021-11-16 14:00"), "label"] = "magnet-partial"
        df.loc[(df["time"] >= "2021-11-16 16:20") & (df["time"] <= "2021-12"), "label"] = "magnet-1" 
        df = df.drop('temperature_surface', 1) #surface temperature is not relevant
        df=df.dropna()
        df=df.drop("time", 1)

        #outlier removal if requested
        if ((self.outlier==1)or(self.outlier==2)):
            self.remove_outlier(df, self.outlier)
        
        #balancing and noise generation. MI DAVA ERRORE IN LOCALE PER VIA DI PROBLEMI NELLE LIBRERIE
        
        if (self.balancing):
          self.data_generation(df)
        self.noise_generation(df) 



        # bearings label creation
        conditions=[
            (df["label"]=="bearings-1") | (df["label"]=="bearings-2"),
            (df["label"]=="working-engine-1") | (df["label"]=="working-engine-2")|(df["label"]=="magnet-partial")|(df["label"]=="magnet-1")
            ]
        values = [1, 0]
        df["bearings"]=np.select(conditions, values)
        #magnet label creation
        conditions=[
            (df["label"]=="magnet-1"),
            (df["label"]=="magnet-partial"),
            (df["label"]=="working-engine-1") | (df["label"]=="working-engine-2")|(df["label"]=="bearings-1")|(df["label"]=="bearings-2")
            ]
        values = [0.7, 0.5, 0]
        df["magnet"]=np.select(conditions, values)
        #creating electricity columns .??? DA SISTEMARE QUANDO CI SONO DATI 
        df["electricity_1"]=0
        df["electricity_2"]=0
        
        #creation of day and machine column
        df["day"]=0
        df["machine"]=0
        df["marked"]=0 #using to signal a 50 measures without variation
        counter_measure=0
        counter_day=0 
        counter_machine=0
        prev_row=0
        marked=1
        #creating random poisson error in label entries
        error_bearings=np.random.poisson(0.001, len(df))
        error_magnets=np.random.poisson(0.001, len(df))
        error_electricity_1=np.random.poisson(0.001, len(df))
        error_electricity_2=np.random.poisson(0.001, len(df))
        final_row=ListDict()
        start_row=ListDict()
        n=0
        prev_row=None
        j=0
        for i, row in df.iterrows():
                if ((prev_row is not None) and ((row.bearings!=prev_row.bearings)or(row.electricity_1!=prev_row.electricity_1)or(row.electricity_2!=prev_row.electricity_2)or(row.magnet!=prev_row.magnet))):#row diverse
                    marked=0
                elif(row.bearings==0):#if all values are same (all defects or no defects at all)
                    marked=-1
                    start_row.add(row)
                    
                if((row.bearings!=0) or (row.electricity_1!=0) or (row.electricity_2!=0) or (row.magnet!=0)):
                    final_row.add(row)
                    
                if (counter_measure==50): #50 measure mark a day
                   
                   counter_measure=0
                   counter_day+=1
                  
                if (counter_day==7):#7 day mark a machine
                  counter_day=0
                  counter_machine+=1
                df.at[i, 'day'] = counter_day
                df.at[i, 'machine'] = counter_machine
                if(error_bearings[n]>0): #given the error add noise to label
                    if(row.bearings==1):
                        df.at[i, 'bearings'] = 0
                    else:
                        df.at[i, 'bearings'] = 1
                elif(error_electricity_1[n]>0):
                    if(row.electricity_1==1):
                        df.at[i, 'electricity_1'] = 0
                    else:
                        df.at[i, 'electricity_1'] = 1
                elif(error_electricity_2[n]>0):
                    if(row.electricity_2==1):
                        df.at[i, 'electricity_2'] = 0
                    else:
                        df.at[i, 'electricity_2'] = 1
                elif(error_magnets[n]>0):
                    if(row.magnet==0.7):
                        df.at[i, 'magnet'] = 0.5
                    elif(row.magnet==0.5):
                        df.at[i, 'magnet'] = 0
                    else:
                        df.at[i, 'magnet'] = 1
    
                if(counter_measure==49):
                    
                    
                    df.at[i, 'marked'] = marked#mark if all defects or no defects in 50 measurements
                    
                    
                    marked=1#reset mark
                else:
                    df.at[i, 'marked'] = 0 #variation of defects
                  
                prev_row=row #aggiorno la row precedente
                counter_measure+=1
                n+=1
                
        df=df.drop("label", 1)


        #correct series of 50 observation without changing in label
        #if there is no defect in 50 series we add a random problem at the end
        #if only defects add no problem at start
        measure_index=0
        first_measure=0
        for i, row in df.iterrows():
            if(measure_index==50):
                measure_index=0
            if(measure_index==0):
                first_measure=i
            if(row.marked==-1):
                rand_row=final_row.get()#obtain casual row 
                for column in list(df.columns):
                    if((column!="machine")and(column!="day")):
                        df.at[i, column]=rand_row[column]
            elif(row.marked==1):
                
                rand_row=start_row.get()
                for column in list(df.columns):
                    if((column!="machine")and(column!="day")):
                        df.at[first_measure, column]=rand_row[column]
        df = df.drop('marked', 1)#drop marked
        df=df.dropna()#dropnull
        
        #in base od dataframe role set resulting dataframe
        if (self.role=="std"):
            df = df.drop('magnet', 1)
        elif(self.role=="magnet"):
            df = df.drop('bearings', 1)
            df = df.drop('electricity_1', 1)
            df = df.drop('electricity_2', 1)

            
        column_list = df['machine'].values.tolist()
        for value in column_list:
            if len(df[(df['machine']==value)]) != 350:
                df=df[df.machine!=value]
        return df
        
    #return part of data when asked
    def return_data(self, epoch, selected=False, splitted=False):

        if selected==True:#if only want selected features
            selected=self.features.selected_features
        else:
            selected=self.features.total_features
        if splitted==False:#if you want splitting in test and train
            return self.df(self.df.day<=epoch)
        else:
              #estraggo il frame per il train
              df_train=self.df[(self.df.day<epoch)]
              self.x_train = df_train[(["day", "machine"]+selected)]
              if(self.role=='magnet'):
                  self.y_train = df_train[(self.features.targets_nb)]
              else:
                  self.y_train = df_train[(self.features.targets)]
              #estraggo i frame per il test
              df_test=self.df[(self.df.day==epoch)]
              self.x_test = df_test[(["day", "machine"]+selected)]
              if(self.role=='magnet'):
                  self.y_test = df_test[(self.features.targets_nb)]
              else:
                  self.y_test = df_test[(self.features.targets)]
              #estraggo per la verifica all epoch successiva
              df_prev=self.df[(self.df.day==(epoch-1))]
              if(self.role=='magnet'):
                  self.y_prev = df_prev[(self.features.targets_nb)]
              else:
                  self.y_prev = df_prev[(self.features.targets)]
              return self.x_train, self.y_train, self.x_test, self.y_test
    #QUESTA SINGOLA FUNZIONE DEVO SISTEMARLA CAMBIANDO DA VALORI DI VITA A GIORNI
    #calculate a dataset for rul prediction
    def calculate_rul_label(self, epoch):
        y=self.df[(self.df.day<=epoch-1)]#take all element before and at that time
        y['censoring']=False
        y['time']=0
        #contatore della misura
        j=0
        #max misura
        max_mes=50*(epoch)#not epoch-1 because epoch is day that start from 0
        #boole var to check if finded failure
        failure=False
        #numeric variable to check how many days before there was failure
        k=0
        
        #ciclo su tutto il df
        for i, row in y.iterrows():
            if(j==max_mes):#arrived to last measurement
                j=0#start from begin
            if(row.magnet==0.7):#so che ha fallito 0.7 E TEMPORANEO FINCHE NON HO GLI 1
                y.at[i, 'censoring'] = True
                failure=True#i found a failure
                k=0
                y.at[i, 'time'] = j+1
            else:
                if failure==False:
                    y.at[i, 'time'] = j+1 #plus one because measurment start from 0
                else:
                    y.at[i, 'time'] = k+1
            j+=1#increase measurement number
            k+=1#increase also how many days after failure
            
            
        y=y[(y.day==epoch-1)]
        y=y[['censoring', 'time']]
        s = y.dtypes
        res = np.array([tuple(x) for x in y.values], dtype=list(zip(s.index, s)))
        return res


#agent for ensemble
class Data_ensemble:
     def __init__(self, models, rul_models, data_operative=None):
         if data_operative is None:#if no data operative object is passed a standard one is used
             self.data_operative=data
         else:
             self.data_operative=data_operative
        
         self.models=models         #obtain number of operative models standard
         self.rul_models=rul_models#obtain rul models 
         
         #obtain values to create X and Y
         self.row_number=self.data_operative.machine_number+1
         self.column_number=50*3*(len(self.models)+2*(len(self.rul_models)))
         self.X_train= np.zeros((self.row_number, self.column_number))
         self.X_train_prev=[]
         self.Y_train_prev=[]
         self.Y_train=np.zeros((self.row_number))
         self.X_test=np.zeros((self.row_number, self.column_number))
         
     def obtain_data(self, epoch, train=True): 
         if(epoch>2):#memorize to do parameter selection in ensemble
             self.X_train_prev=self.X_train
             self.Y_train_prev=self.Y_train
         if (train==True):           
            self.X_train=np.zeros((self.row_number, self.column_number))
            error_y=np.random.poisson(0.01, self.row_number)
            self.Y_train=np.zeros((self.row_number))
            
            n_model=0
            k=0
            for model in self.models:#for eache standard model
                n_label=0
                for label_pred in model.prev_pred:#for each problem
                    j=(0+n_model*150+n_label*50)#column number
                    i=0#row number
                    for pred in label_pred:
                        if ((j%50)==0):
                            j=0+n_model*150+n_label*50
                            i+=1
                        self.X_train[i][j]=pred      
                        j+=1
                        k+=1
                    n_label+=1
                n_model+=1
            
            n_rul_model=0
            for model in self.rul_models:#for each rul models
                    j=(0+n_model*150+n_rul_model*100+n_label*50)#column number
                    i=0#row number
                    n_label=0
                    for pred in model.prev_pred:
                        if ((j%50)==0):
                            j=0+n_model*150+n_rul_model*300+n_label*50
                            i+=1#increase machine
                        self.X_train[i][j]=pred      
                        j+=1
                    n_label+=1
                    j=0+n_model*150+n_rul_model*100+n_label*50
                    for pred in model.prev_pred_RUL:
                        if ((j%50)==0):
                            j=0+n_model*150+n_rul_model*100+n_label*50
                            i+=1#increase machine
                        self.X_train[i][j]=pred      
                        j+=1
                    n_rul_model+=1
            self.X_train = pd.DataFrame(self.X_test)

            for label in self.data._operative.features.targets:
                df=self.data_operative.df[self.data_operative.df.day==epoch]#riduco il dataframe all'epoca che stiamo valutando
                i=0
                j=0
                for elem in df[label]:
                    if (i==50):
                        i=0
                        j+=1
                    
                    if((elem==1)and(k==0)):
                        if(error_y[j]==0):
                            self.Y_train[j]=1

                    elif(elem==1)and(k!=0)and(label != "magnet"):
                        if(error_y[j]==0):
                            self.Y_train[j]+=0.5
                            
                    elif(elem!=0)and(label == "magnet"):
                        if(error_y[j]==0):
                            self.Y_train[j]+=elem
                    i+=1
                    
            
                k+=1
                
            #polish label
            for entry in self.Y_train:
                if (entry<1):
                    entry=0
                elif (entry>1):
                    entry=1
            #update total features column
            self.features=Features(selected=list(self.X_train.columns))
            
            

         else:#for test porpouse
            self.X_test=np.zeros((self.row_number, self.column_number))
            n_model=0
            k=0
            for model in self.models:
                n_label=0
                for label_pred in model.last_pred:
                    j=(0+n_model*150+n_label*50)
                    i=0
                    for pred in label_pred:
                        if ((j%50)==0):
                            j=0+n_model*150+n_label*50
                            i+=1
                        self.X_test[i][j]=pred      
                        j+=1
                        k+=1
                    n_label+=1
                n_model+=1
            
            n_rul_model=0
            for model in self.rul_models:
                    j=(0+n_model*150+n_rul_model*100+n_label*50)
                    i=0#row number
                    n_label=0
                    for pred in model.last_pred_RUL:
                        if ((j%50)==0):
                            j=0+n_model*150+n_rul_model*300+n_label*50
                            i+=1#increase machine
                        self.X_test[i][j]=pred      
                        j+=1
                    n_label+=1
                    j=0+n_model*150+n_rul_model*100+n_label*50
                    for pred in model.last_pred_RUL:
                        if ((j%50)==0):
                            j=0+n_model*150+n_rul_model*100+n_label*50
                            i+=1#increase machine
                        self.X_test[i][j]=pred      
                        j+=1
                    n_rul_model+=1
            self.X_test = pd.DataFrame(self.X_test)



#VARIABLE
data=Data_operative(role="general")  #an operative data frame that has a general lookup capacibility
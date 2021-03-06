# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 22:02:26 2021

@author: MyPc
"""

# LIBRARIES
import data_manipulation  # our data_mainpulation library
import numpy as np
import pandas as pd
import math
from math import sqrt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import brier_score_loss
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt

# CLASSES


# operative agent
class operative_agent_model:
    def __init__(
        self,
        name,
        outlier=1,
        balancing=True,
        role=None,
        dataset=None,
        targets=None,
        epoch=None,
    ):

        # base information
        self.outlier = outlier  # outlier elimination
        self.balancing = balancing  # balancing of dataset
        self.name = name  # name of agent classifier

        # setting epoch
        if epoch is None:
            self.epoch = 2
        else:  # default case
            self.epoch = epoch

        # dealing with role
        if role is None:
            self.role = "std"

        elif role == "magnet":
            self.role = "magnet"
        else:  # default case
            self.role = role

        # setting classifier
        if name == "logistic":
            self.model = LogisticRegression(random_state=0)
            self.C = 1.0
            self.C_list = [0.0001, 0.001, 0.1, 1.0, 100]
            self.penalty = "none"
            self.penalty_list = ["l2", "l1"]
            self.tol = 1e-3
            self.weight = None
            self.weight_list = ["balanced", None]
            self.model.set_params(tol=self.tol)
            self.solver = "liblinear"
            self.model.set_params(solver=self.solver)
            self.probability = True  # indica se il modello predice anche probabilit??
        elif name == "svm":
            self.model = svm.SVC(probability=True)
            self.C = 1.0
            self.C_list = [0.0001, 0.001, 0.1, 1.0, 100]
            self.kernel = "rbf"
            self.kernel_list = ["rbf", "linear", "sigmoid"]
            self.gamma = "scale"
            self.gamma_list = ["scale", "auto", 0.0001, 0.001, 0.1, 1.0, 100]
            self.shrinking = True
            self.shrinking_list = [True, False]
            self.tol = 1e-3
            self.model.set_params(tol=self.tol)
            self.class_weight = None
            self.class_weight_list = ["balanced", None]
            self.probability = True
            if self.role == "magnet":
                self.probability = False

        elif name == "knn":
            self.model = KNeighborsClassifier()
            self.n_neighbors = 5
            self.n_neighbors_list = list(range(5, 60))
            self.weights = "uniform"
            self.weights_list = ["uniform", "distance"]
            self.metric = "euclidean"
            self.metric_list = ["euclidean", "chebyshev", "manhattan"]
            self.probability = False
        elif name == "lda":
            self.model = LinearDiscriminantAnalysis()
            self.solver = "lsqr"
            self.model.set_params(solver=self.solver)
            self.shrinkage = None
            self.shrinkage_list = ["auto", None]
            self.probability = True
        elif name == "ridge":
            self.model = RidgeClassifier(random_state=0)
            self.alpha = 1.0
            self.alpha_list = [0.0001, 0.001, 0.1, 1.0, 100]
            self.tol = 1e-3
            self.model.set_params(tol=self.tol)
            self.class_weight = None
            self.class_weight_list = [None, "balanced"]
            self.probability = False

        else:  # default case
            self.model = LogisticRegression(random_state=0)
            self.probability = False

        if role == "magnet":
            self.ovo_model = OneVsOneClassifier(
                self.model
            )  # wrap in onve vs one classifier

        print("created agent " + self.name + " role " + self.role)

        if (
            dataset is None
        ):  # dealing with dataset agent connected with our operative agent on base of role
            if self.role == "std":
                self.dataset = data_manipulation.Data_operative(
                    self.outlier, self.balancing
                )

            elif self.role == "magnet":
                self.dataset = data_manipulation.Data_operative(
                    self.outlier, self.balancing, self.role
                )
                self.auxiliary = auxiliary_rul_agent_model(self.dataset)

            else:
                self.dataset = data_manipulation.Data_operative(
                    self.outlier, self.balancing, self.role
                )
        else:  # default case
            self.dataset = dataset

        print("created dataset of agent " + self.name + " role " + self.role)

        # a priori values
        self.n = self.dataset.machine_number

        # scaler
        self.scaler_list = [StandardScaler(), StandardScaler(), StandardScaler()]
        self.scaler = StandardScaler()  # used only for rul predictors

        # in caso ?? magnet
        # only for RUL
        self.prev_pred_RUL = []
        self.last_pred_RUL = []

        # evaluation metrics

        # stored results
        self.matrix_list = []  # confusion matrices of previous inspection
        self.f1_pred = 0  # f1 previous inspection
        self.last_pred = []  # ultima previsione
        self.last_pred_proba = []  # probability last inspection
        self.prev_pred_proba = []  # probability previous inspection
        self.prev_pred = []  # previous prediction
        self.prev_label = []  # labels of previous inspection

        # model evalutation metrics
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.accuracy = 0
        self.corr_coef = 0
        self.brier = 0  # brier score
        self.pr_rec = []  # precision-recal score
        self.perc_effort = 100  # effort of inspection
        self.ROC = []
        self.total_score = 0
        self.cl_error = 0

    # feature selection
    def feature_select(self, epoch):
        max_f1 = 0
        feature_selected = []
        feature_obtained = []
        if self.role == "std":
            if (
                len(self.dataset.features.total_features) <= 4
            ):  # if number of features is lower than 4 do all combination
                for features_subset in self.dataset.features.subsets():
                    f1_subset = 0

                    clf = self.model

                    # model preparation
                    clf = self.model
                    i = 0
                    f1_subset = 0
                    for label in self.dataset.features.targets:

                        f1_subset += self.cross_validate(
                            clf,
                            (["day", "machine"] + features_subset),
                            self.total_score_calculation,
                            label,
                            self.scaler_subset[i],
                            matrix=True,
                        )
                        i += 1
                    f1_subset = f1_subset / 3

                    if f1_subset > max_f1:
                        max_f1 = f1_subset
                        feature_selected = features_subset
            else:  # bottom up if combinatorial explosion
                sel_feat = set()
                df_train = self.dataset.df[(self.dataset.df.day < (epoch))]
                X = df_train[
                    (["day", "machine"] + self.dataset.features.total_features)
                ]
                y = df_train[self.dataset.features.targets]
                n = 0  # number of features estimated
                corr = X.corr(method="kendall")
                corr.fillna(0)
                total = corr.to_numpy().sum()
                corr = total / (len(corr.columns.values) * len(corr))
                if corr == 0:
                    n = min(int(len(X) / 2), len(X.columns.values))
                elif corr > 0:
                    n = min(int(len(X) - 1), len(X.columns.values))
                else:
                    n = min(int(sqrt(len(X))), len(X.columns.values))
                for label in self.dataset.features.targets:
                    clf = ExtraTreesClassifier(n_estimators=100)
                    clf = clf.fit(X, y[label])
                    model = SelectFromModel(clf, prefit=True)
                    X_new = X.iloc[:, model.get_support(indices=True)]
                    sel_feat.update(list(X_new.columns.values))
                feature_obtained = list(sel_feat)

                if len(feature_obtained) < n:
                    for label in self.dataset.features.targets:
                        if len(y[label].unique()) != 1:
                            lsvc = LinearSVC().fit(X, y[label])
                            model = SelectFromModel(lsvc, prefit=True)
                            X_new = X.iloc[:, model.get_support(indices=True)]
                            sel_feat.update(list(X_new.columns.values))
                    feature_obtained = list(sel_feat)

                if len(feature_obtained) < n:
                    for label in self.dataset.features.targets:
                        model = SelectKBest(f_classif, k=n)
                        X_new = model.fit_transform(X, y[label])
                        X_new = X.iloc[:, model.get_support(indices=True)]
                        sel_feat.update(list(X_new.columns.values))
                    feature_obtained = list(sel_feat)
                features_subset = []
                for elem in feature_obtained:
                    features_subset.append(elem)

                    # model preparation
                    clf = self.model
                    i = 0
                    f1_subset = 0
                    for label in self.dataset.features.targets:

                        f1_subset += self.cross_validate(
                            clf,
                            (["day", "machine"] + features_subset),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )
                        i += 1
                    f1_subset = f1_subset / 3

                    if f1_subset > max_f1:
                        max_f1 = f1_subset
                        feature_selected = features_subset

            self.dataset.features.selected_features = feature_selected

        else:  # magnet case
            if len(self.dataset.features.total_features) <= 4:
                for features_subset in self.dataset.features.subsets():
                    f1_subset = 0
                    clf = OneVsOneClassifier(self.model)
                    f1_subset = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.features_subset),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                    )
                    if f1_subset > max_f1:
                        max_f1 = f1_subset
                        feature_selected = features_subset
            else:
                sel_feat = set()
                df_train = self.dataset.df[(self.dataset.df.day < (epoch))]
                X = df_train[
                    (["day", "machine"] + self.dataset.features.total_features)
                ]
                y = df_train[self.dataset.features.targets_binarized]
                n = 0  # number of features estimated
                corr = X.corr(method="kendall")
                corr.fillna(0)
                total = corr.to_numpy().sum()
                corr = total / (len(corr.columns.values) * len(corr))
                if corr == 0:
                    n = min(int(len(X) / 2), len(X.columns.values))
                elif corr > 0:
                    n = min(int(len(X) - 1), len(X.columns.values))
                else:
                    n = min(int(sqrt(len(X))), len(X.columns.values))
                for label in self.dataset.features.targets_binarized:
                    clf = ExtraTreesClassifier(n_estimators=100)
                    clf = clf.fit(X, y[label])
                    model = SelectFromModel(clf, prefit=True)
                    X_new = X.iloc[:, model.get_support(indices=True)]
                    sel_feat.update(list(X_new.columns.values))
                feature_obtained = list(sel_feat)
                if len(feature_obtained) < n:
                    for label in self.dataset.features.targets_binarized:
                        if len(y[label].unique()) != 1:
                            lsvc = LinearSVC().fit(X, y[label])
                            model = SelectFromModel(lsvc, prefit=True)
                            X_new = X.iloc[:, model.get_support(indices=True)]
                            sel_feat.update(list(X_new.columns.values))
                    feature_obtained = list(sel_feat)

                if len(feature_obtained) < n:
                    for label in self.dataset.features.targets_binarized:
                        model = SelectKBest(f_classif, k=n)
                        X_new = model.fit_transform(X, y[label])
                        X_new = X.iloc[:, model.get_support(indices=True)]
                        sel_feat.update(list(X_new.columns.values))
                    feature_obtained = list(sel_feat)
                features_subset = []
                for elem in feature_obtained:
                    features_subset.append(elem)
                    f1_subset = 0

                    clf = OneVsOneClassifier(self.model)
                    f1_subset = self.cross_validate(
                        clf,
                        (["day", "machine"] + features_subset),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )
                    if f1_subset > max_f1:
                        max_f1 = f1_subset
                        feature_selected = features_subset

            self.dataset.features.selected_features = feature_selected

    # parameter selection
    def parameter_selection(self, epoch):
        if self.role == "std":  # for standard case
            if self.name == "knn":
                max_f1 = 0
                matrix_list = []
                for param in self.n_neighbors_list:
                    f1_param = 0
                    # model prepration
                    clf = self.model
                    clf.set_params(n_neighbors=param)
                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3
                    if f1_param > max_f1:
                        self.n_neighbors = param
                        max_f1 = f1_param
                self.model.set_params(n_neighbors=self.n_neighbors)

                max_f1 = 0
                matrix_list = []
                for param in self.weights_list:
                    f1_param = 0
                    # model prepration
                    clf = self.model
                    clf.set_params(weights=param)
                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3
                    if f1_param > max_f1:
                        self.weights = param
                        max_f1 = f1_param
                self.model.set_params(weights=self.weights)

                max_f1 = 0
                matrix_list = []
                for param in self.metric_list:
                    f1_param = 0
                    # model prepration
                    clf = self.model
                    clf.set_params(metric=param)

                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3
                    if f1_param > max_f1:
                        self.metric = param
                        max_f1 = f1_param
                self.model.set_params(metric=self.metric)

            elif self.name == "lda":

                max_f1 = 0
                matrix_list = []
                for param in self.shrinkage_list:
                    f1_param = 0
                    clf = self.model
                    clf.set_params(shrinkage=param)
                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3
                    if f1_param > max_f1:
                        self.shrinkage = param
                        max_f1 = f1_param
                self.model.set_params(shrinkage=self.shrinkage)
                max_f1 = 0
                matrix_list = []
            elif self.name == "logistic":
                max_f1 = 0
                matrix_list = []
                for param in self.penalty_list:
                    f1_param = 0
                    clf = self.model
                    clf.set_params(penalty=param)
                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3
                    if f1_param > max_f1:
                        self.penalty = param
                        max_f1 = f1_param
                self.model.set_params(penalty=self.penalty)
                max_f1 = 0
                matrix_list = []

                for param in self.C_list:
                    f1_param = 0
                    clf = self.model
                    clf.set_params(C=param)
                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3
                    if f1_param > max_f1:
                        self.C = param
                        max_f1 = f1_param
                self.model.set_params(C=self.C)
                max_f1 = 0
                matrix_list = []

                for param in self.weight_list:
                    f1_param = 0
                    i = 0
                    clf = self.model
                    clf.set_params(class_weight=param)
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3
                    if f1_param > max_f1:
                        self.weight = param
                        max_f1 = f1_param
                self.model.set_params(class_weight=self.weight)
                max_f1 = 0
                matrix_list = []

            elif self.name == "svm":

                max_f1 = 0
                matrix_list = []
                for param in self.kernel_list:
                    f1_param = 0

                    clf = self.model
                    clf.set_params(kernel=param)
                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3

                    if f1_param > max_f1:
                        self.kernel = param
                        max_f1 = f1_param
                self.model.set_params(kernel=self.kernel)
                max_f1 = 0
                matrix_list = []

                for param in self.C_list:
                    f1_param = 0
                    clf = self.model
                    clf.set_params(C=param)
                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3

                    if f1_param > max_f1:
                        self.C = param
                        max_f1 = f1_param
                self.model.set_params(C=self.C)
                max_f1 = 0
                matrix_list = []

                for param in self.gamma_list:
                    f1_param = 0
                    clf = self.model
                    clf.set_params(gamma=param)
                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3

                    if f1_param > max_f1:
                        self.gamma = param
                        max_f1 = f1_param
                self.model.set_params(gamma=self.gamma)
                max_f1 = 0
                matrix_list = []

                for param in self.shrinking_list:
                    f1_param = 0

                    clf = self.model
                    clf.set_params(shrinking=param)
                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3

                    if f1_param > max_f1:
                        self.shrinking = param
                        max_f1 = f1_param
                self.model.set_params(shrinking=self.shrinking)
                max_f1 = 0
                matrix_list = []

                for param in self.class_weight_list:
                    f1_param = 0

                    clf = self.model
                    clf.set_params(class_weight=param)
                    i = 0
                    for label in self.dataset.features.targets:
                        f1_param += self.cross_validate(
                            clf,
                            (
                                ["day", "machine"]
                                + self.dataset.features.selected_features
                            ),
                            self.total_score_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=True,
                        )

                        i += 1
                    f1_param = f1_param / 3

                    if f1_param > max_f1:
                        self.class_weight = param
                        max_f1 = f1_param
                self.model.set_params(class_weight=self.class_weight)
                max_f1 = 0
                matrix_list = []
        else:  # magnet case
            if self.name == "knn":
                max_f1 = 0
                matrix_list = []
                for param in self.n_neighbors_list:
                    f1_param = 0

                    clf = self.model
                    clf.set_params(n_neighbors=param)
                    clf = OneVsOneClassifier(self.model)
                    f1_param = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.dataset.features.selected_features),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )

                    if f1_param > max_f1:
                        self.n_neighbors = param
                        max_f1 = f1_param
                self.model.set_params(n_neighbors=self.n_neighbors)

                max_f1 = 0
                matrix_list = []
                for param in self.weights_list:
                    f1_param = 0

                    clf = self.model
                    clf.set_params(weights=param)
                    clf = OneVsOneClassifier(self.model)
                    f1_param = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.dataset.features.selected_features),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )

                    if f1_param > max_f1:
                        self.weights = param
                        max_f1 = f1_param
                self.model.set_params(weights=self.weights)

                max_f1 = 0
                matrix_list = []
                for param in self.metric_list:
                    f1_param = 0

                    clf = self.model
                    clf.set_params(metric=param)
                    clf = OneVsOneClassifier(self.model)
                    f1_param = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.dataset.features.selected_features),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )

                    if f1_param > max_f1:
                        self.metric = param
                        max_f1 = f1_param
                self.model.set_params(metric=self.metric)

            elif self.name == "svm":

                max_f1 = 0
                matrix_list = []
                for param in self.kernel_list:
                    f1_param = 0

                    clf = self.model
                    clf.set_params(kernel=param)
                    clf = OneVsOneClassifier(self.model)
                    f1_param = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.dataset.features.selected_features),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )
                    if f1_param > max_f1:
                        self.kernel = param
                        max_f1 = f1_param
                self.model.set_params(kernel=self.kernel)
                max_f1 = 0
                matrix_list = []

                for param in self.C_list:
                    f1_param = 0

                    clf = self.model
                    clf.set_params(C=param)
                    clf = OneVsOneClassifier(self.model)
                    f1_param = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.dataset.features.selected_features),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )
                    if f1_param > max_f1:
                        self.C = param
                        max_f1 = f1_param
                self.model.set_params(C=self.C)
                max_f1 = 0
                matrix_list = []

                for param in self.gamma_list:
                    f1_param = 0

                    clf = self.model
                    clf.set_params(gamma=param)
                    clf = OneVsOneClassifier(self.model)
                    f1_param = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.dataset.features.selected_features),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )
                    if f1_param > max_f1:
                        self.gamma = param
                        max_f1 = f1_param
                self.model.set_params(gamma=self.gamma)
                max_f1 = 0
                matrix_list = []

                for param in self.shrinking_list:
                    f1_param = 0
                    clf = self.model
                    clf.set_params(shrinking=param)
                    clf = OneVsOneClassifier(self.model)

                    f1_param = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.dataset.features.selected_features),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )
                    if f1_param > max_f1:
                        self.shrinking = param
                        max_f1 = f1_param
                self.model.set_params(shrinking=self.shrinking)
                max_f1 = 0
                matrix_list = []

                for param in self.class_weight_list:
                    f1_param = 0

                    clf = self.model
                    clf.set_params(class_weight=param)
                    clf = OneVsOneClassifier(self.model)
                    f1_param = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.dataset.features.selected_features),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )

                    if f1_param > max_f1:
                        self.class_weight = param
                        max_f1 = f1_param
                self.model.set_params(class_weight=self.class_weight)
                max_f1 = 0
                matrix_list = []
            elif self.name == "ridge":
                max_f1 = 0
                matrix_list = []

                for param in self.alpha_list:
                    f1_param = 0
                    clf = self.model
                    clf.set_params(alpha=param)
                    clf = OneVsOneClassifier(self.model)
                    f1_param = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.dataset.features.selected_features),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )
                    if f1_param > max_f1:
                        self.alpha = param
                        max_f1 = f1_param
                self.model.set_params(alpha=self.alpha)
                max_f1 = 0
                matrix_list = []

                for param in self.class_weight_list:
                    f1_param = 0
                    clf = self.model
                    clf.set_params(class_weight=param)
                    clf = OneVsOneClassifier(self.model)
                    f1_param = self.cross_validate(
                        clf,
                        (["day", "machine"] + self.dataset.features.selected_features),
                        self.total_score_calculation,
                        self.dataset.features.targets_nb,
                        self.scaler,
                        matrix=True,
                    )
                    if f1_param > max_f1:
                        self.class_weight = param
                        max_f1 = f1_param
                self.model.set_params(class_weight=self.class_weight)
                max_f1 = 0
                matrix_list = []

    def cl_error_calculation(self, matrix_list, single=False):
        if single:
            matr = matrix_list
            numerator = 0
            total = 0

            total += matr[1][0] + matr[0][1] + matr[1][1] + matr[0][0]
            numerator += matr[1][1] + matr[0][0]

        else:
            numerator = 0
            total = 0
            for matr in matrix_list:
                total += matr[1][0] + matr[0][1] + matr[1][1] + matr[0][0]
                numerator += matr[1][1] + matr[0][0]

        return 1 - ((numerator / total))

    # score and metrics functions
    def total_score_calculation(self, matrix_list, single=False):
        cl_error = self.cl_error_calculation(matrix_list, single)
        confidence = self.calculate_confidence(cl_error)
        f1_calculation = self.f1_calculation(matrix_list, single)
        return confidence + f1_calculation

    def f1_calculation(self, matrix_list, single=False):
        if single:
            matr = matrix_list
            tp = 0
            fn = 0
            fp = 0

            fp += matr[1][0]
            fn += matr[0][1]
            tp += matr[0][0]
        else:
            tp = 0
            fn = 0
            fp = 0
            for matr in matrix_list:
                fp += matr[1][0]
                fn += matr[0][1]
                tp += matr[0][0]
        if (2 * tp + fn + fp) == 0:
            denominator = 0.000001
        else:
            denominator = 2 * tp + fn + fp
        return tp / denominator

    # t is setted with 26 machines  should be changed if increase
    def calculate_confidence(self, error, t=1.676, n=50):
        return t * sqrt((error * (1 - error)) / n)

    def recall_calculation(self, matrix_list, single=False):
        if single:
            matr = matrix_list
            tp = 0
            fn = 0

            fn += matr[0][1]
            tp += matr[0][0]
        else:
            tp = 0
            fn = 0
            for matr in matrix_list:
                fn += matr[0][1]
                tp += matr[0][0]
        return tp / (tp + fn)

    def precision_calculation(self, matrix_list, single=False):
        if single:
            matr = matrix_list
            tp = 0
            fp = 0

            fp += matr[1][0]
            tp += matr[0][0]
        else:
            tp = 0
            fp = 0
            for matr in matrix_list:
                fp += matr[1][0]
                tp += matr[0][0]
        return tp / (tp + fp)

    def accuracy_calculation(self, matrix_list, single=False):
        if single:
            matr = matrix_list
            tp = 0
            tn = 0
            nn = 0
            pp = 0

            tn += matr[1][1]
            tp += matr[0][0]
            nn += matr[1][1] + matr[1][0]
            pp += matr[0][0] + matr[0][1]

            TPR = tp / pp
            TNR = tn / nn
        else:
            tp = 0
            tn = 0
            nn = 0
            pp = 0
            for matr in matrix_list:
                tn += matr[1][1]
                tp += matr[0][0]
                nn += matr[1][1] + matr[1][0]
                pp += matr[0][0] + matr[0][1]

            TPR = tp / pp
            TNR = tn / nn

        return TPR + TNR / 2

    def corr_coef_calculation(self, matrix_list, single=False):
        if single:
            matr = matrix_list
            tp = 0
            fn = 0
            fp = 0
            tn = 0

            tn += matr[1][1]
            tp += matr[0][0]
            fn += matr[0][1]
            fp += matr[1][0]

            num = tp * tn - fp * fn
            den_sqrt = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        else:
            tp = 0
            fn = 0
            fp = 0
            tn = 0
            for matr in matrix_list:
                tn += matr[1][1]
                tp += matr[0][0]
                fn += matr[0][1]
                fp += matr[1][0]

            num = tp * tn - fp * fn
            den_sqrt = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if den_sqrt > 0:
            den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        else:
            den = 0.000001
        return num / den

    def roc_calculation(self, y, score, single=False, pos_label=2):
        roc_list = []
        if single:
            fpr, tpr, thresholds = metrics.roc_curve(y, score, pos_label)
            lista = []
            lista.append(fpr)
            lista.append(tpr)
            lista.append(thresholds)
            roc_list.append(lista)
        else:
            for i in range(0, len(score)):
                fpr, tpr, thresholds = metrics.roc_curve(y[i], score[i], pos_label)
                lista = []
                lista.append(fpr)
                lista.append(tpr)
                lista.append(thresholds)
                roc_list.append(lista)
        return roc_list

    def pr_rec_calculation(self, y, score, single=False):
        prc_list = []
        if single:
            pr, rec, thr = metrics.roc_curve(y, score)
            lista = []
            lista.append(pr)
            lista.append(rec)
            lista.append(thr)
            prc_list.append(lista)
        else:
            for i in range(0, len(score)):
                pr, rec, thr = metrics.roc_curve(y[i], score[i])
                lista = []
                lista.append(pr)
                lista.append(rec)
                lista.append(thr)
                prc_list.append(lista)
        return prc_list

    def brier_calculation(self, y, score, single=False):
        brier = 0
        if single:
            brier = brier_score_loss(y, score)
        else:
            for i in range(0, len(score)):
                brier += brier_score_loss(y[i], score[i])
                brier = brier / 3
        return brier

    def perc_effort_calculation(self, matrix_list, single=False, multiplicative=False):
        if multiplicative == False:
            multiplicative = 1
        else:
            multiplicative = 100

        if single:
            matr = matrix_list
            real_effort = 0
            actual_effort = 0
            tn = 0
            tp = 0
            fn = 0
            fp = 0
            tn += matr[1][1]
            tp += matr[0][0]
            fn += matr[0][1]
            fp += matr[1][0]
            real_effort = tn + tp + fn + fp
            actual_effort = tp + fp
            returnvalue = ((real_effort - actual_effort) / real_effort) * multiplicative
        else:
            real_effort = 0
            actual_effort = 0
            tn = 0
            tp = 0
            fn = 0
            fp = 0
            for matr in matrix_list:
                tn += matr[1][1]
                tp += matr[0][0]
                fn += matr[0][1]
                fp += matr[1][0]
            real_effort = tn + tp + fn + fp
            actual_effort = tp + fp
            returnvalue = ((real_effort - actual_effort) / real_effort) * multiplicative
            returnvalue = returnvalue / 3

        return returnvalue

    # call al metrics calculation
    def calculate_metrics(self):

        clf = self.model

        features = ["day", "machine"] + self.dataset.features.selected_features
        if self.role == "magnet":
            target = self.dataset.features.targets_nb
            clf = OneVsOneClassifier(clf)

        else:
            target = self.dataset.features.targets

        if self.role == "magnet":
            self.f1 = self.cross_validate(
                clf,
                features,
                self.f1_calculation,
                target,
                self.scaler,
                matrix=True,
                score=False,
            )
            self.precision = self.cross_validate(
                clf,
                features,
                self.precision_calculation,
                target,
                self.scaler,
                matrix=True,
                score=False,
            )
            self.recall = self.cross_validate(
                clf,
                features,
                self.recall_calculation,
                target,
                self.scaler,
                matrix=True,
                score=False,
            )
            self.accuracy = self.cross_validate(
                clf,
                features,
                self.accuracy_calculation,
                target,
                self.scaler,
                matrix=True,
                score=False,
            )
            self.corr_coef = self.cross_validate(
                clf,
                features,
                self.corr_coef_calculation,
                target,
                self.scaler,
                matrix=True,
                score=False,
            )
            self.perc_effort = self.cross_validate(
                clf,
                features,
                self.perc_effort_calculation,
                target,
                self.scaler,
                matrix=True,
                score=False,
            )

            if (
                self.probability == True
            ):  # metrics that could be calculated only if probabilities are included
                self.ROC = self.cross_validate(
                    clf,
                    features,
                    self.roc_calculation,
                    target,
                    self.scaler,
                    matrix=False,
                    score=True,
                    list_out=True,
                )
                self.pr_rec = self.cross_validate(
                    clf,
                    features,
                    self.pr_rec_calculation,
                    target,
                    self.scaler,
                    matrix=False,
                    score=True,
                    list_out=True,
                )
                self.brier = self.cross_validate(
                    clf,
                    features,
                    self.brier_calculation,
                    target,
                    self.scaler,
                    matrix=False,
                    score=True,
                )

        else:
            i = 0
            for label in target:
                self.f1 += self.cross_validate(
                    clf,
                    features,
                    self.f1_calculation,
                    label,
                    self.scaler_list[i],
                    matrix=True,
                    score=False,
                )
                i += 1
            self.f1 = self.f1 / 3
            i = 0
            for label in target:
                self.precision += self.cross_validate(
                    clf,
                    features,
                    self.precision_calculation,
                    label,
                    self.scaler_list[i],
                    matrix=True,
                    score=False,
                )
                i += 1
            self.precision = self.precision / 3
            i = 0
            for label in target:
                self.recall += self.cross_validate(
                    clf,
                    features,
                    self.recall_calculation,
                    label,
                    self.scaler_list[i],
                    matrix=True,
                    score=False,
                )
                i += 1
            self.recall = self.recall / 3
            i = 0
            for label in target:
                self.accuracy += self.cross_validate(
                    clf,
                    features,
                    self.accuracy_calculation,
                    label,
                    self.scaler_list[i],
                    matrix=True,
                    score=False,
                )
                i += 1
            self.accuracy = self.accuracy / 3
            i = 0
            for label in target:
                self.corr_coef += self.cross_validate(
                    clf,
                    features,
                    self.corr_coef_calculation,
                    label,
                    self.scaler_list[i],
                    matrix=True,
                    score=False,
                )
                i += 1
            self.corr_coef = self.corr_coef / 3
            i = 0
            for label in target:
                self.perc_effort += self.cross_validate(
                    clf,
                    features,
                    self.perc_effort_calculation,
                    label,
                    self.scaler_list[i],
                    matrix=True,
                    score=False,
                )
                i += 1
            self.perc_effort = self.perc_effort / 3

            i = 0
            if (
                self.probability == True
            ):  # metrics that could be calculated only if probabilities are included
                for label in target:
                    self.ROC.append(
                        self.cross_validate(
                            clf,
                            features,
                            self.roc_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=False,
                            score=True,
                            list_out=True,
                        )
                    )
                    i += 1
                i = 0
                for label in target:
                    self.pr_rec.append(
                        self.cross_validate(
                            clf,
                            features,
                            self.pr_rec_calculation,
                            label,
                            self.scaler_list[i],
                            matrix=False,
                            score=True,
                            list_out=True,
                        )
                    )
                    i += 1
                i = 0
                for label in target:
                    self.brier += self.cross_validate(
                        clf,
                        features,
                        self.brier_calculation,
                        label,
                        self.scaler_list[i],
                        matrix=False,
                        score=True,
                    )
                    i += 1
                self.brier = self.brier / 3

        return 0

    # call all performance functions
    def performance_calculation(self):

        clf = self.model

        if self.role == "std":

            i = 0
            for label in self.dataset.features.targets:

                self.total_score += self.cross_validate(
                    clf,
                    (["day", "machine"] + self.dataset.features.selected_features),
                    self.total_score_calculation,
                    label,
                    self.scaler_list[i],
                    matrix=True,
                )

                i += 1
            self.total_score = self.total_score / 3
            i = 0
            for label in self.dataset.features.targets:

                self.cl_error += self.cross_validate(
                    clf,
                    (["day", "machine"] + self.dataset.features.selected_features),
                    self.cl_error_calculation,
                    label,
                    self.scaler_list[i],
                    matrix=True,
                )

                i += 1
            self.cl_error = self.cl_error / 3
            self.confidence = self.calculate_confidence(self.cl_error)

        else:  # magnet
            # preparazione modello
            clf = OneVsOneClassifier(self.model)
            self.total_score = self.cross_validate(
                clf,
                (["day", "machine"] + self.dataset.features.selected_features),
                self.total_score_calculation,
                self.dataset.features.targets_nb,
                self.scaler,
                matrix=True,
            )
            self.cl_error = self.cross_validate(
                clf,
                (["day", "machine"] + self.dataset.features.selected_features),
                self.cl_error_calculation,
                self.dataset.features.targets_nb,
                self.scaler,
                matrix=True,
            )
            self.confidence = self.calculate_confidence(self.cl_error)

        self.calculate_metrics()

    # model trainign
    def train_model(self):
        if self.role == "std":  # standard role
            if self.last_pred:  # if a prevision already exists
                self.prev_pred = []
                self.prev_label = []
                # saving precedents label
                for label in self.dataset.features.targets:
                    self.prev_label.append(self.dataset.y_prev[label])
                # saving precedent previsions
                for sublist in self.last_pred:
                    self.prev_pred.append(sublist)
                if self.probability == True:
                    self.prev_pred_proba = []
                    for sublist in self.last_pred_proba:
                        self.prev_pred_proba.append(sublist)
                    self.last_pred_proba = []
                # confusion matrix calculation
                self.matrix_list = []
                for i in range(0, 3):
                    self.matrix_list.append(
                        confusion_matrix(
                            self.prev_label[i], self.prev_pred[i], labels=[1, 0]
                        )
                    )
                # f1 score calculation
                self.f1_pred = self.total_score_calculation(self.matrix_list)
                self.last_pred = []
                # train
                i = 0
                for label in self.dataset.features.targets:
                    if len(self.dataset.y_train[label].unique()) != 1:
                        self.scaler_list[i].fit_transform(self.dataset.x_train)
                        self.model.fit(
                            self.dataset.x_train[
                                self.dataset.features.selected_features
                            ],
                            self.dataset.y_train[label],
                        )
                        i += 1
            else:  # trick in case not previous labels
                i = 0
                n = 2500
                for label in self.dataset.features.targets:

                    if len(self.dataset.y_train[label].unique()) != 1:
                        self.scaler_list[i].fit_transform(self.dataset.x_train)
                        self.model.fit(
                            self.dataset.x_train, self.dataset.y_train[label]
                        )
                        y_pred1 = self.model.predict(self.dataset.x_train)
                        self.matrix_list.append(
                            confusion_matrix(
                                self.dataset.y_train[label], y_pred1, labels=[1, 0]
                            )
                        )
                        y_pred = y_pred1[np.arange(0, n)]
                        y_train = self.dataset.y_train[label].iloc[:-n]
                        self.prev_label.append((y_train).values)
                        self.prev_pred.append(y_pred)
                    else:  # if labels are always same perfect prevision suposed
                        y_train = self.dataset.y_train[label].iloc[:-n]
                        self.prev_label.append((y_train).values)
                        self.prev_pred.append((y_train).values)
                    i += 1
                self.f1_pred = self.total_score_calculation(self.matrix_list)
        else:  # magnet case
            if len(self.last_pred) > 0:
                self.prev_pred = []
                self.prev_label = []
                self.prev_label = self.dataset.y_prev
                self.prev_pred = self.last_pred
                if self.probability == True:
                    self.prev_pred_proba = []
                    self.prev_pred_proba = self.last_pred_proba
                    self.last_pred_proba = []
                self.matrix_list = []
                self.matrix_list = multilabel_confusion_matrix(
                    self.prev_label, self.prev_pred, labels=[1, 0]
                )
                self.f1_pred = self.total_score_calculation(self.matrix_list)
                self.last_pred = []
                self.prev__pred_RUL = self.last_pred_RUL
                self.last_pred_RUL = []
                self.scaler.fit_transform(self.dataset.x_train)
                self.ovo_model.fit(self.dataset.x_train, self.dataset.y_train)
            else:
                self.scaler.fit_transform(self.dataset.x_train)
                self.ovo_model.fit(self.dataset.x_train, self.dataset.y_train)
                y_pred1 = self.ovo_model.predict(self.dataset.x_train)
                self.matrix_list = multilabel_confusion_matrix(
                    self.dataset.y_train, y_pred1, labels=[1, 0]
                )
                n = int(len(self.dataset.x_train) / 2)
                y_pred = y_pred1[np.arange(0, n)]
                self.auxiliary.add(y_pred.tolist())
                y_train = self.dataset.df[(self.dataset.df.day == 1)]
                y_train = y_train[self.dataset.features.targets_nb]

                self.prev_label = y_train
                self.prev_pred = y_pred
                self.f1_pred = self.total_score_calculation(self.matrix_list)

    # timeseries crossvalidation function
    def cross_validate(
        self,
        clf,
        features,
        metric,
        target,
        scaler=None,
        matrix=False,
        score=False,
        list_out=False,
    ):
        value = 0
        result = 0
        single = True
        if self.role == "magnet":
            single = False
        if list_out:
            value = []

        for day in range(
            1, self.epoch
        ):  # E SE ANDASSE CAMBIATO ANCHE ENSEMBLE A EPOCH E NON EPOCH-1

            # dataframe extraction for train
            x_train = self.dataset.df[
                (self.dataset.df.day < (day))
            ]  # NON SO SE QUESTO VADA CAMBIATO ANCHE NELL ENSEMBLE
            x_train = x_train[features]
            y_train = self.dataset.df[(self.dataset.df.day < (day))]
            y_train = y_train[target]
            # dataframe extraction for test
            x_test = self.dataset.df[(self.dataset.df.day == (day))]
            x_test = x_test[features]
            y_test = self.dataset.df[(self.dataset.df.day == (day))]
            y_test = y_test[target]

            if scaler is None:
                self.scaler.fit_transform(x_train)
                self.scaler.fit_transform(x_test)
            else:
                scaler.fit_transform(x_train)
                scaler.fit_transform(x_test)
            # train and classification
            if single:
                if len(y_train.unique()) != 1:
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
                else:
                    y_pred = y_test
            else:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
            if self.role == "std":
                matr = confusion_matrix(y_test, y_pred, labels=[1, 0])
            else:
                matr = multilabel_confusion_matrix(y_test, y_pred, labels=[1, 0])

            if score:
                if not list_out:
                    score_cal = clf.predict_proba(x_test)[:, 1]
                    value += metric(y_test, score_cal, single)
                    result = value / (self.epoch - 1)
                else:
                    score_cal = clf.predict_proba(x_test)[:, 1]
                    value.append(metric(y_test, score_cal, single))
                    result = value
            elif matrix:
                value += metric(matr, single)
                result = value / (self.epoch - 1)
            else:
                value += metric(y_pred, y_test, single)
                result = value / (self.epoch - 1)

        return result

        # scaler selection

    def scaler_selection(self, epoch):
        scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler()]
        selected_normalizer = []
        max_f1 = 0
        if self.role == "std":  # for standard case
            for scaler1 in scalers:
                for scaler2 in scalers:
                    for scaler3 in scalers:
                        scaler_subset = [scaler1, scaler2, scaler3]
                        # model preparation
                        clf = self.model
                        i = 0
                        f1_subset = 0
                        for label in self.dataset.features.targets:
                            f1_subset += self.cross_validate(
                                clf,
                                (
                                    ["day", "machine"]
                                    + self.dataset.features.selected_features
                                ),
                                self.total_score_calculation,
                                label,
                                scaler_subset[i],
                                matrix=True,
                            )

                            i += 1
                        f1_subset = f1_subset / 3
                        if f1_subset > max_f1:
                            selected_normalizer = scaler_subset
                            max_f1 = f1_subset

            self.scaler_list = selected_normalizer

        else:  # for magnet
            for scaler in scalers:
                # preparazione modello
                clf = OneVsOneClassifier(self.model)
                f1_subset = self.cross_validate(
                    clf,
                    (["day", "machine"] + self.dataset.features.selected_features),
                    self.total_score_calculation,
                    self.dataset.features.targets_nb,
                    self.scaler,
                    matrix=True,
                )
                if f1_subset > max_f1:
                    selected_normalizer = scaler
                    max_f1 = f1_subset

            self.scaler = selected_normalizer

        return 0

    # model testing/prediction
    def test_model(self):
        matrix_list = []
        if self.role == "std":  # if standard
            i = 0
            for label in self.dataset.features.targets:
                self.scaler_list[i].fit_transform(self.dataset.x_test)  # scaling
                # prediction
                self.last_pred.append(self.model.predict(self.dataset.x_test))
                # if classification has probability calculate them
                if self.probability == True:
                    self.last_pred_proba = self.model.predict_proba(self.dataset.x_test)
                if self.epoch > 2:
                    matrix_list.append(
                        confusion_matrix(self.prev_label[i], self.prev_pred[i])
                    )
                i += 1
            self.f1_pred = self.f1_calculation(self.matrix_list)
        else:  # magnet
            self.scaler.fit_transform(self.dataset.x_test)
            self.last_pred = self.ovo_model.predict(self.dataset.x_test)
            if self.probability == True:
                self.last_pred_proba = self.ovo_model.predict_proba(self.dataset.x_test)
            if self.epoch > 2:
                matrix_list = multilabel_confusion_matrix(
                    self.prev_label, self.prev_pred
                )
            self.f1_pred = self.f1_calculation(matrix_list)

    # single operative cycle step
    def operative_cycle(self):
        self.dataset.return_data(self.epoch, False, True)
        self.feature_select(self.epoch)  # find best features
        self.parameter_selection(self.epoch)  # find best parameters
        self.scaler_selection(self.epoch)  # find best scalres
        if (
            self.role == "std"
        ):  # if standard role set classifier with founded hyperparameter
            if self.name == "logistic":
                self.model = LogisticRegression(
                    C=self.C,
                    penalty=self.penalty,
                    class_weight=self.weight,
                    solver=self.solver,
                    random_state=0,
                )
            elif self.name == "svm":
                self.model = svm.SVC(
                    C=self.C,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    shrinking=self.shrinking,
                    probability=True,
                    tol=self.tol,
                    class_weight=self.class_weight,
                )
            elif self.name == "knn":
                self.model = KNeighborsClassifier(
                    n_neighbors=self.n_neighbors,
                    weights=self.weights,
                    metric=self.metric,
                )
            elif self.name == "lda":
                self.model = LinearDiscriminantAnalysis(
                    solver=self.solver, shrinkage=self.shrinkage
                )
        elif self.role == "magnet":  # if magnet wrap classiffier after setted
            if self.name == "ridge":
                self.model = OneVsOneClassifier(
                    RidgeClassifier(
                        alpha=self.alpha,
                        tol=self.tol,
                        class_weight=self.class_weight,
                        random_state=0,
                    )
                )
            elif self.name == "svm":
                self.model = OneVsOneClassifier(
                    svm.SVC(
                        C=self.C,
                        kernel=self.kernel,
                        gamma=self.gamma,
                        shrinking=self.shrinking,
                        tol=self.tol,
                        class_weight=self.class_weight,
                    )
                )
            elif self.name == "knn":
                self.model = OneVsOneClassifier(
                    KNeighborsClassifier(
                        n_neighbors=self.n_neighbors,
                        weights=self.weights,
                        metric=self.metric,
                    )
                )
        self.dataset.return_data(self.epoch, True, True)  # obtain new datas
        self.train_model()  # train model
        self.performance_calculation()  # obtain performances on training
        self.test_model()  # do prediction
        if self.role == "magnet":
            self.last_pred_RUL = self.auxiliary.RUL_prediction(
                self.epoch, self.last_pred
            )
            if self.epoch == 2:
                self.prev_pred_RUL = self.last_pred_RUL

        # reset models
        if self.role == "std":  # if standard role
            if self.name == "logistic":
                self.model = LogisticRegression(
                    tol=self.tol, solver=self.solver, random_state=0
                )
            elif self.name == "svm":
                self.model = svm.SVC(probability=True)
            elif self.name == "knn":
                self.model = KNeighborsClassifier()
            elif self.name == "lda":
                self.model = LinearDiscriminantAnalysis()
                self.model.set_params(solver=self.solver)
        elif self.role == "magnet":  # if magnet wrap classiffier after setted
            if self.name == "ridge":
                self.ovo_model = OneVsOneClassifier(RidgeClassifier(tol=self.tol))
                self.model = RidgeClassifier(tol=self.tol)
            elif self.name == "svm":
                self.ovo_model = OneVsOneClassifier(svm.SVC(probability=True))
                self.model = svm.SVC(probability=True)

            elif self.name == "knn":
                self.ovo_model = OneVsOneClassifier(KNeighborsClassifier())
                self.model = KNeighborsClassifier()

        print(
            "prediction done "
            + self.name
            + " role "
            + self.role
            + " epoch "
            + str(self.epoch)
        )
        self.epoch += 1  # increase epoch

    # full operative cycle simulation
    def operative_cycle_sim(self):
        self.epoch = 2
        for epoch in range(self.epoch, 7):  # full operative cycle
            self.operative_cycle()


# auxiliary RUL agent
class auxiliary_rul_agent_model:
    def __init__(self, dataset, name=None):
        self.dataset = dataset

        if name is None:  # set random survival forest as default rul predictor
            self.name = "RandomSurvivalForest"
            self.model = RandomSurvivalForest(n_estimators=100, random_state=0)

        else:
            self.name = name

        self.rul_X = (
            []
        )  # contains all prediction till now of classic operative OVO classifier
        self.last_pred = []  # last prediction risk quantity
        self.surv = []  # survival function
        self.hazard = []  # cumulative hazard

    def add(self, OVOpred):
        self.rul_X.append(OVOpred)

    def RUL_prediction(self, epoch, OVOpred):
        dummy_rul = self.rul_X[:]
        dummy_rul.append(dummy_rul[len(dummy_rul) - 1])
        X_train = pd.DataFrame.from_records(dummy_rul)
        X_train = X_train.T
        self.add(OVOpred)
        X_test = pd.DataFrame.from_records(self.rul_X)
        X_test = X_test.T
        y_train = self.dataset.calculate_rul_label(epoch)  # obtain rul label
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        self.model.fit(X_train, y_train)
        self.last_pred = self.model.predict(X_test)
        self.surv = self.model.predict_survival_function(X_test, return_array=True)
        self.hazard = self.model.predict_cumulative_hazard_function(
            X_test, return_array=True
        )
        return self.last_pred

    # depict survival of last prediction
    def depict_survival(self):
        if self.surv is not None:
            for i, s in enumerate(self.surv):
                plt.step(self.model.event_times_, s, where="post", label=str(i))
                plt.ylabel("Survival probability")
                plt.xlabel("Time in days")
                plt.legend()
                plt.grid(True)

    # depict survival of last prediction
    def depict_hazard(self):
        for i, s in enumerate(self.hazard):
            plt.step(self.model.event_times_, s, where="post", label=str(i))
            plt.ylabel("Cumulative hazard")
            plt.xlabel("Time in days")
            plt.legend()
            plt.grid(True)

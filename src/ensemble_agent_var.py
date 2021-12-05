# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 22:03:08 2021

@author: MyPc
"""
from forest_variation import RandomForestClassifier_variation
from operative_agent_var import operative_agent_model
from sklearn.ensemble import RandomForestClassifier
import operative_agent_var as operative_agent
import data_manipulation
from sklearn.metrics import confusion_matrix
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
import forest_variation
from sklearn import metrics
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif


class ensemble_agent_model:
    def __init__(
        self,
        epoch=None,
        dataset=None,
        model=None,
        model_list=None,
        rul_model_list=None,
        opt_function=None,
    ):
        # set epoch
        if epoch is None:
            self.epoch = 2
        else:
            self.epoch = epoch
        # set role
        self.role = "ensemble"
        # set list of model contained
        if model_list is None:
            self.model_list = [
                operative_agent.operative_agent_model("knn", role="std"),
                operative_agent.operative_agent_model("logistic", role="std"),
                operative_agent.operative_agent_model("lda", role="std"),
                # operative_agent.operative_agent_model("svm", role="std"),
            ]
            self.rul_model_list = [
                # operative_agent.operative_agent_model("knn", role="magnet"),
                operative_agent.operative_agent_model("svm", role="magnet"),
                operative_agent.operative_agent_model("ridge", role="magnet"),
            ]
        else:
            for elem in model_list:
                self.model_list.append(operative_agent_model(elem, role="std"))
            for elem in rul_model_list:
                self.rul_model_list.append(operative_agent_model(elem, role="magnet"))
        # set dataset
        if dataset is None:
            self.dataset = data_manipulation.Data_ensemble(
                self.model_list, self.rul_model_list
            )
        else:
            self.dataset = dataset
        # set model
        if model is None:
            self.model = RandomForestClassifier_variation(random_state=0)
            self.name = "RandomForestClassifier Variation"
        elif model == "RandomForest":
            self.model = RandomForestClassifier(random_state=0)
            self.name = "RandomForestClassifier"
        else:
            self.model = model

        # fixed parameter
        self.n = self.dataset.row_number  # number of considered machines

        print("created agent " + self.name + " role " + self.role)

        # value used to fine tune modl

        # parameters
        self.n_estimators = 1  # THIS NUMBER MUST BE CHECKED??
        self.estimator_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.criterion = "gini"
        self.criterion_list = ["gini", "entropy"]
        self.min_sample_split = 0.01
        self.min_sample_split_list = [1.0, 0.5, 0.25, 0.1, 0.01]
        self.min_sample_leaf = 1
        self.min_sample_leaf_list = [1, 5, 10, 20, 40, 60]
        self.bootstrap = False
        self.bootstrap_list = [False, True]
        self.max_features = "sqrt"
        self.max_features_list = ["sqrt"]

        # scaler
        self.scaler = StandardScaler()

        # evaluation metrics

        # stored results
        # this results are used to evaluate actual run
        self.last_pred = []  # last prediction values
        self.last_pred_proba = []  # last prediction probabilities
        self.prev_pred = []  # previous  prediction
        self.prev_pred_proba = []  # previous prediction probabilities
        self.confusion_matrix = 0  # confusion matrix of last prediction
        self.f1_pred = 0  # f1 previous prediction
        self.savings = 0  # saving untill now

        # model evaluation metrix
        # this are values used for evaluate model capabilities in general
        self.confidence = 0  # prediction confidence
        self.cl_error = 0
        self.f1 = 0
        self.precision = 0
        self.recall = 0
        self.accuracy = 0
        self.corr_coef = 0
        self.brier = 0
        self.pr_rec = []
        self.perc_effort = 100
        self.ROC = []

    # functions used to calculate metrics

    def cl_error_calculation(self, y_pred, y):
        numerator = 0
        total = 0
        matr = confusion_matrix(y, y_pred, labels=[1, 0])
        total += matr[1][0] + matr[0][1] + matr[1][1] + matr[0][0]
        numerator += matr[1][1] + matr[0][0]
        return 1 - ((numerator / total))

    def calculate_confidence(self, error, t=1.676, n=50):
        return t * sqrt((error * (1 - error)) / n)

    def recall_calculation(self, matr):
        tp = 0
        fn = 0
        fn += matr[0][1]
        tp += matr[0][0]
        return tp / (tp + fn)

    def precision_calculation(self, matr):
        tp = 0
        fp = 0
        fp += matr[1][0]
        tp += matr[0][0]
        return tp / (tp + fp)

    def accuracy_calculation(self, matr):
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
        return TPR + TNR / 2

    def corr_coef_calculation(self, matr):
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        tn += matr[1][1]
        tp += matr[0][0]
        fn += matr[0][1]
        fp += matr[1][0]

        num = tp * tn - fp * fn
        den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return num / den

    def roc_calculation(self, y, score, pos_label=2):
        roc_list = []
        fpr, tpr, thresholds = metrics.roc_curve(y, score, pos_label)
        lista = []
        lista.append(fpr)
        lista.append(tpr)
        lista.append(thresholds)
        roc_list.append(lista)
        return roc_list

    def pr_rec_calculation(self, y, score):
        prc_list = []
        pr, rec, thr = metrics.roc_curve(y, score)
        lista = []
        lista.append(pr)
        lista.append(rec)
        lista.append(thr)
        prc_list.append(lista)
        return prc_list

    def brier_calculation(self, y, score):
        brier = brier_score_loss(y, score)
        return brier

    def perc_effort_calculation(self, matr, multiplicative=False):
        if multiplicative == False:
            multiplicative = 1
        else:
            multiplicative = 100
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

        return ((real_effort - actual_effort) / real_effort) * multiplicative

    def f1_calculation(self, matr):
        tp = 0
        fn = 0
        fp = 0
        fp += matr[1][0]
        fn += matr[0][1]
        tp += matr[0][0]
        return tp / (2 * tp + fn + fp)

    # function to calculate metrics
    def calculate_metrics(self):
        clf = self.model
        self.f1 = self.cross_validate(
            clf,
            self.dataset.data_operative.features.selected_features,
            self.f1_calculation,
            scaler=None,
            matrix=True,
            score=False,
        )
        self.precision = self.cross_validate(
            clf,
            self.dataset.data_operative.features.selected_features,
            self.precision_calculation,
            scaler=None,
            matrix=True,
            score=False,
        )
        self.recall = self.cross_validate(
            clf,
            self.dataset.data_operative.features.selected_features,
            self.recall_calculation,
            scaler=None,
            matrix=True,
            score=False,
        )
        self.accuracy = self.cross_validate(
            clf,
            self.dataset.data_operative.features.selected_features,
            self.accuracy_calculation,
            scaler=None,
            matrix=True,
            score=False,
        )
        self.corr_coef = self.cross_validate(
            clf,
            self.dataset.data_operative.features.selected_features,
            self.corr_coef_calculation,
            scaler=None,
            matrix=True,
            score=False,
        )
        self.perc_effort = self.cross_validate(
            clf,
            self.dataset.data_operative.features.selected_features,
            self.perc_effort_calculation,
            scaler=None,
            matrix=True,
            score=False,
        )
        self.ROC = self.cross_validate(
            clf,
            self.dataset.data_operative.features.selected_features,
            self.roc_calculation,
            scaler=None,
            matrix=False,
            score=True,
            list_out=True,
        )
        self.pr_rec = self.cross_validate(
            clf,
            self.dataset.data_operative.features.selected_features,
            self.pr_rec_calculation,
            scaler=None,
            matrix=False,
            score=True,
            list_out=True,
        )
        self.brier = self.cross_validate(
            clf,
            self.dataset.data_operative.features.selected_features,
            self.brier_calculation,
            scaler=None,
            matrix=False,
            score=True,
        )

    # function to set performances
    def performance_calculation(self):
        # model preparation
        clf = self.model
        self.cl_error = self.cross_validate(
            clf,
            self.dataset.data_operative.features.selected_features,
            self.cl_error_calculation,
            scaler=None,
            score=False,
        )
        self.confidence = self.calculate_confidence(self.cl_error)
        self.calculate_metrics()

    # scaler selection
    def scaler_selection(self):
        scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler()]
        max_save = 0
        for scaler in scalers:

            clf = self.model
            save_scaler = self.cross_validate(
                clf,
                self.dataset.data_operative.features.selected_features,
                forest_variation.save_calculation,
                scaler,
            )
            if save_scaler > max_save:
                max_save = save_scaler
                self.scaler = scaler
        return 0

    # timeseries crossvalidation function
    def cross_validate(
        self,
        clf,
        features,
        metric,
        scaler=None,
        matrix=False,
        score=False,
        list_out=False,
    ):
        value = 0
        result = 0
        if list_out:
            value = []
        for day in range(1, self.epoch - 1):
            # dataframe extraction for train
            x_train = self.dataset.X_train_prev[day - 1]
            x_train = x_train[features]
            y_train = self.dataset.Y_train_prev[day - 1]
            # dataframe extraction for test
            if day < self.epoch - 2:
                x_test = self.dataset.X_train_prev[day]
                x_test = x_test[features]
                y_test = self.dataset.Y_train_prev[day]
            else:
                x_test = self.dataset.X_train[features]
                y_test = self.dataset.Y_train
            if scaler is None:
                self.scaler.fit_transform(x_train)
                self.scaler.fit_transform(x_test)
            else:
                scaler.fit_transform(x_train)
                scaler.fit_transform(x_test)
            # train and classification
            clf.fit(x_train, y_train.values.ravel())
            y_pred = clf.predict(x_test)
            matr = confusion_matrix(y_test, y_pred, labels=[1, 0])

            if score:
                if not list_out:
                    score_cal = clf.predict_proba(x_test)[:, 1]
                    value += metric(y_test, score_cal)
                    result = value / (self.epoch - 1)
                else:
                    score_cal = clf.predict_proba(x_test)[:, 1]
                    value.append(metric(y_test, score_cal))
                    result = value
            elif matrix:
                value += metric(matr)
                result = value / (self.epoch - 1)
            else:
                value += metric(y_pred, y_test)
                result = value / (self.epoch - 1)

        return result

    def parameter_selection(self):
        max_save = 0
        for parameter in self.estimator_list:
            clf = self.model
            clf.set_params(n_estimators=parameter)
            save_parameter = self.cross_validate(
                clf,
                self.dataset.data_operative.features.selected_features,
                forest_variation.save_calculation,
            )
            if save_parameter > max_save:
                max_save = save_parameter
                self.n_estimator = parameter

        self.model.set_params(n_estimators=self.n_estimators)

        max_save = 0
        for parameter in self.min_sample_split_list:
            clf = self.model
            clf.set_params(min_samples_split=parameter)
            save_parameter = self.cross_validate(
                clf,
                self.dataset.data_operative.features.selected_features,
                forest_variation.save_calculation,
            )
            if save_parameter > max_save:
                max_save = save_parameter
                self.min_sample_split = parameter

        self.model.set_params(min_samples_split=self.min_sample_split)

        max_save = 0
        for parameter in self.min_sample_leaf_list:
            clf = self.model
            clf.set_params(min_samples_leaf=parameter)
            save_parameter = self.cross_validate(
                clf,
                self.dataset.data_operative.features.selected_features,
                forest_variation.save_calculation,
            )
            if save_parameter > max_save:
                max_save = save_parameter
                self.min_sample_leaf = parameter

        self.model.set_params(min_samples_leaf=self.min_sample_leaf)

        max_save = 0
        for parameter in self.bootstrap_list:
            clf = self.model
            clf.set_params(bootstrap=parameter)
            save_parameter = self.cross_validate(
                clf,
                self.dataset.data_operative.features.selected_features,
                forest_variation.save_calculation,
            )
            if save_parameter > max_save:
                max_save = save_parameter
                self.bootstrap = parameter

        self.model.set_params(bootstrap=self.bootstrap)

        max_save = 0
        for parameter in self.max_features_list:
            clf = self.model
            clf.set_params(max_features=parameter)
            save_parameter = self.cross_validate(
                clf,
                self.dataset.data_operative.features.selected_features,
                forest_variation.save_calculation,
            )
            if save_parameter > max_save:
                max_save = save_parameter
                self.max_features = parameter

        self.model.set_params(max_features=self.max_features)

        max_save = 0
        for parameter in self.criterion_list:
            clf = self.model
            clf.set_params(criterion=parameter)
            save_parameter = self.cross_validate(
                clf,
                self.dataset.data_operative.features.selected_features,
                forest_variation.save_calculation,
            )
            if save_parameter > max_save:
                max_save = save_parameter
                self.criterion = parameter

        self.model.set_params(criterion=self.criterion)

        return 0

    # feature selection function
    def feature_selection(self):

        max_save = 0
        feature_selected = []
        if len(list(self.dataset.X_train.columns)) <= 4:  # combinatorial approach
            for features_subset in data_manipulation.data.features.subsets(
                list(self.dataset.X_train_prev.columns)
            ):
                save_subset = 0
                # model preparation
                clf = self.model
                save_subset = self.cross_validate(
                    clf, features_subset, forest_variation.save_calculation
                )
                if save_subset > max_save:
                    max_save = save_subset
                    feature_selected = features_subset
        else:  # bottom up approach
            feature_set = []
            X = self.dataset.X_train
            y = self.dataset.Y_train
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

            clf = ExtraTreesClassifier(n_estimators=100)  # SETTED TO DEFAULT VALUES
            clf = clf.fit(X, y)
            model = SelectFromModel(clf, prefit=True)
            X_new = X.iloc[:, model.get_support(indices=True)]
            feature_set = list(X_new.columns.values)
            if len(feature_set) < n:
                X = self.dataset.X_train
                y = self.dataset.Y_train
                lsvc = LinearSVC(C=0.1).fit(X, y)
                model = SelectFromModel(lsvc, prefit=True)
                X_new = X.iloc[:, model.get_support(indices=True)]
                feature_set = list(X_new.columns.values)
            if len(feature_set) < n:
                X = self.dataset.X_train
                y = self.dataset.Y_train
                model = SelectKBest(f_classif, k="7")
                X_new = model.fit_transform(X, y)
                X_new = X.iloc[:, model.get_support(indices=True)]
                feature_set = list(X_new.columns.values)

            feature_subset = []
            for elem in feature_set:
                feature_subset.append(elem)
                # model preparation
                clf = self.model
                save_subset = self.cross_validate(
                    clf, feature_subset, forest_variation.save_calculation
                )
                if save_subset > max_save:
                    max_save = save_subset
                    feature_selected = feature_subset

        if len(feature_selected) < min(int(sqrt(len(X))), len(X.columns.values)):
            model = SelectKBest(f_classif, k=n)
            X_new = model.fit_transform(self.dataset.X_train, self.dataset.Y_train)
            X_new = X.iloc[:, model.get_support(indices=True)]
            feature_selected = list(X_new.columns.values)

        if len(feature_selected) > (n / 2):
            self.max_features_list.append(int(sqrt(len(feature_selected)) / 2))
            self.max_features_list.append(int(sqrt(len(feature_selected)) / 4))
            self.max_features_list.append(int(sqrt(len(feature_selected)) / 6))
        else:
            self.max_features_list.append(
                min(int(sqrt(len(feature_selected)) * 2)), len(X.columns.values)
            )
            self.max_features_list.append(
                min(int(sqrt(len(feature_selected)) * 4)), len(X.columns.values)
            )
            self.max_features_list.append(
                min(int(sqrt(len(feature_selected)) * 6)), len(X.columns.values)
            )

        if 0 in self.max_features_list:
            self.max_features_list.remove(0)

        self.dataset.data_operative.features.selected_features = feature_selected

    # train model
    def train_model(self):
        self.feature_selection()  # if no train set do however a prior features selection
        self.parameter_selection()
        if self.name == "RandomForestClassifier":
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                min_samples_split=self.min_sample_split,
                min_samples_leaf=self.min_sample_leaf,
                bootstrap=self.bootstrap,
                max_features=self.max_features,
                random_state=0,
            )  # set classifier founded parameter
        else:
            self.model = RandomForestClassifier_variation(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                min_samples_split=self.min_sample_split,
                min_samples_leaf=self.min_sample_leaf,
                bootstrap=self.bootstrap,
                max_features=self.max_features,
                random_state=0,
            )  # set classifier founded parameter
        self.scaler_selection()

        self.performance_calculation()  # calculate performance to be expected for next prediction
        self.scaler.fit_transform(
            self.dataset.X_train
        )  # do a transformation of features befor training
        if self.name == "RandomForestClassifier":
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                min_samples_split=self.min_sample_split,
                min_samples_leaf=self.min_sample_leaf,
                bootstrap=self.bootstrap,
                max_features=self.max_features,
                random_state=0,
            )  # set classifier founded parameter
        else:
            self.model = RandomForestClassifier_variation(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                min_samples_split=self.min_sample_split,
                min_samples_leaf=self.min_sample_leaf,
                bootstrap=self.bootstrap,
                max_features=self.max_features,
                random_state=0,
            )  # set classifier founded parameter
        self.model.fit(
            self.dataset.X_train[
                self.dataset.data_operative.features.selected_features
            ],
            self.dataset.Y_train.values.ravel(),
        )  # do train

    # testing model
    def test_model(self):

        self.scaler.fit_transform(self.dataset.X_test)
        # save previous pred
        self.prev_pred = self.last_pred
        self.prev_pred_proba = self.last_pred_proba
        if self.epoch > 2:
            self.confusion_matrix = confusion_matrix(
                self.dataset.Y_train, self.prev_pred
            )
            self.fi_pred = self.f1_calculation(self.confusion_matrix)
            self.savings += forest_variation.save_calculation(
                self.prev_pred, self.dataset.Y_train
            )
        self.last_pred = self.model.predict(
            self.dataset.X_test[self.dataset.data_operative.features.selected_features]
        )  # do and save prediction
        self.last_pred_proba = self.model.predict_proba(
            self.dataset.X_test[self.dataset.data_operative.features.selected_features]
        )  # do the same for probabilities

    # operative cycle simulation
    def operative_cycle(self):
        for (
            model
        ) in self.model_list:  # call an operative cycle for all models inside ensemble
            model.operative_cycle()
        for model in self.rul_model_list:
            model.operative_cycle()

        self.dataset.obtain_data(self.epoch, train=True)  # obtain dataset for training
        self.train_model()  # train model
        self.dataset.obtain_data(self.epoch, train=False)  # obtain dataset for testing
        self.test_model()  # test model

        if self.name == "RandomForestClassifier":
            self.model = RandomForestClassifier(random_state=0)  # reset original model
        else:
            self.model = RandomForestClassifier_variation(random_state=0)
        self.max_features_list = ["sqrt"]
        print(
            "prediction done "
            + self.name
            + " role "
            + self.role
            + " epoch "
            + str(self.epoch)
        )
        self.epoch += 1  # increase epoch


ensemble = ensemble_agent_model()

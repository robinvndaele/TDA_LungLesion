__author__ = 'Robin Vandaele'

import numpy as np # handling arrays and general math
import pandas as pd # working with data frames
import warnings # ignore warnings
import itertools # cross product of lists
from pymrmr import mRMR # minimum redundancy maximum relevance feature selection
from sklearn.utils._testing import ignore_warnings # ignore sklearn warnings
from sklearn.exceptions import ConvergenceWarning # specify sklearn warning to avoid being printed
from sklearn.base import BaseEstimator, TransformerMixin # base functions for building custom sklearn transformers
from sklearn.impute import SimpleImputer # for imputing missing values
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, KBinsDiscretizer # preprocessing features
from sklearn.compose import ColumnTransformer # transform selected columns (e.g. numeric/categorical only)
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression # (Bayesian) linear models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier # random forest models
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier # k-nearest-neighbors models
from sklearn.svm import SVC, SVR # support vector machine models
from sklearn.naive_bayes import GaussianNB # Gaussian naive bayes classification model
from sklearn.ensemble import VotingRegressor, StackingRegressor, VotingClassifier, StackingClassifier # ensemble models
from sklearn.pipeline import make_pipeline # make pipeline combining preprocessing and ML models
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score # testing multiple models
from xgboost import XGBRegressor, XGBClassifier # XGBoost models

# NOTE: unlike the SF-PA data, the feature evualation functions here specifically target the LIDC data.
# This is because the folds for cross validation are obtained from the patients, not their nodules.
# Matching the patients to their nodules is assumed to be possible through the row indices of our data.


class mrmrSelector(BaseEstimator, TransformerMixin):
    # Class Constructor 
    def __init__(self, n_bins=5, n_features=10, method="MIQ"):
        super().__init__()
        self.cols = None
        self.discretizer = None
        self.n_bins = n_bins
        self.n_features = n_features
        self.method = method
    
    def fit(self, X, y):
        self.discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="uniform").fit(X)
        temp = pd.DataFrame(np.column_stack([y, self.discretizer.transform(X)]))
        temp.columns = [str(i) for i in range(temp.shape[1])]
        self.cols =  [int(i) - 1 for i in mRMR(temp, self.method, self.n_features)]
        
        return self
        
    def transform(self, X):
        
        return self.discretizer.transform(X)[:,self.cols]


@ignore_warnings(category=ConvergenceWarning)
def run_exps_reg(X_dict: dict, y: pd.DataFrame, X_base=None, X_uni=None, stratify=None, n_splits=5, n_repeats=10, metric="r2", random_state=None):
    '''
    Lightweight script to test many models and find winners
    :param X_dict: dict of featured DataFrames named by featured types  
    :param y: target vector
    :param X_base: a baseline type of (preprocessed) features that will be included for evaluation, but not for concatenation/ensemble models
    :param X_uni: universal features that will be concatenated for every model (e.g., a contrast label)
    :param stratify: binary vector that specifies how the patients should be stratified (not the outcome per nodule) 
    :param n_splits: int specifying the number of folds
    :metric: String specifying the scoring function
    :random_state: int, RandomState instance or None, to control the randomness of the experiments
    :return: a tuple containing a DataFrame with all performances per fold, and a dictionary containing all built models
    '''

    warnings.simplefilter("ignore", category=UserWarning)
    
    results = {}
    
    numeric_transformer = make_pipeline(SimpleImputer(), MinMaxScaler(), mrmrSelector())
    categorical_transformer = OneHotEncoder()
    
    X = pd.concat([X_dict[feature_type] for feature_type in X_dict.keys()] + [X_uni], axis=1)
    
    models = [
        ("LR", LinearRegression()), 
        ("RF", RandomForestRegressor(random_state=random_state)),
        ("KNN", KNeighborsRegressor()),
        ("SVM", SVR(kernel="linear")), 
        ("BR", BayesianRidge()),
        ("XGB", XGBRegressor(disable_default_eval_metric=True, random_state=random_state))
    ]
    
    pidx = dict()
    pidx_to_pnodidxs = dict()

    for i, pnod in enumerate(X.index):
        p, nod = pnod.split("_")
        if p not in pidx.keys():
            pidx[p] = len(pidx.keys())
            pidx_to_pnodidxs[pidx[p]] = list()
        pidx_to_pnodidxs[pidx[p]].append(i)

    cv = list()
    if stratify is None:
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for train_index, test_index in rkf.split(range(len(pidx)), stratify):
        pnod_train_index = np.hstack([pidx_to_pnodidxs[idx] for idx in train_index])
        pnod_test_index = np.hstack([pidx_to_pnodidxs[idx] for idx in test_index])
        cv.append((pnod_train_index, pnod_test_index))
    del rkf, pidx, pidx_to_pnodidxs
    
    final = pd.DataFrame()
    all_estimators = dict()
    
    for model_name, model in models:

        all_estimators[model_name] = dict()
        
        print("Calculating performances for model: {}".format(model_name))
        
        estimators = list()
        
        # seperate models
        
        for feature_type in ["base"] + list(X_dict.keys()):

            if feature_type == "base":
                if X_base is None:
                    continue
                else:
                    this_X = X_base
                    this_model = model
                    all_estimators[model_name][feature_type] = this_model

            else:
                this_X = pd.concat([X_dict[feature_type], X_uni], axis=1)
                numeric_features = this_X._get_numeric_data().columns
                categorical_features = this_X.columns.difference(numeric_features)
        
                preprocessor = ColumnTransformer(transformers=[
                    ("num", numeric_transformer, numeric_features), 
                    ("cat", categorical_transformer, categorical_features)
                ], remainder="drop")

                this_model = make_pipeline(preprocessor, model)
                estimators.append((model_name + "_" + feature_type, this_model))
                all_estimators[model_name][feature_type] = this_model
            
            this_scores = pd.DataFrame(cross_val_score(this_model, this_X, y, scoring=metric, cv=cv), columns=["score"])
            this_scores["model"] = model_name
            this_scores["type"] = feature_type
            final = pd.concat([final, this_scores], axis=0)
            
        # concatenated features
        
        numeric_features = X._get_numeric_data().columns
        categorical_features = X.columns.difference(numeric_features)
        
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features), 
            ("cat", categorical_transformer, categorical_features)
        ], remainder="drop")
        
        this_model = make_pipeline(preprocessor, model)
        all_estimators[model_name]["concat"] = this_model
        this_scores = pd.DataFrame(cross_val_score(this_model, X, y, scoring=metric, cv=cv), columns=["score"])
        this_scores["model"] = model_name
        this_scores["type"] = "concat"
        final = pd.concat([final, this_scores], axis=0)
            
        # ensemble models
            
        ensembles = [
            ("vote_soft", VotingRegressor(estimators=estimators)),
            ("stack", StackingRegressor(estimators=estimators, final_estimator=model))
        ]
        
        for ensemble_name, ensemble in ensembles:
            
            all_estimators[model_name][ensemble_name] = ensemble
            this_scores = pd.DataFrame(cross_val_score(ensemble, X, y, scoring=metric, cv=cv), columns=["score"])
            this_scores["model"] = model_name
            this_scores["type"] = ensemble_name
            final = pd.concat([final, this_scores], axis=0)
    
    final.index = range(final.shape[0])

    final_summary = pd.DataFrame()

    models = pd.unique(final["model"])
    types = pd.unique(final["type"])

    for t in types:
        It = pd.Series(final["type"] == t)
        for m in models:
            I = np.where(It & pd.Series(final["model"] == m))[0]
            final_summary.loc[m, t + " score"] = np.mean(final.loc[I, "score"])
            final_summary.loc[m, t + " std"] = np.std(final.loc[I, "score"])
        final_summary.loc["mean", t + " std"] = np.std(final.loc[np.where(It)[0], "score"])

    for c in final_summary.columns:
        if c.endswith("std"):
            continue
        final_summary.loc["mean", c] = np.nanmean(final_summary.loc[:, c])

    return final, all_estimators, final_summary


@ignore_warnings(category=ConvergenceWarning)
def run_exps_class(X_dict: dict, y: pd.DataFrame, X_base=None, X_uni=None, stratify=None, n_splits=5, n_repeats=10, metric="roc_auc", random_state=None):
    '''
    Lightweight script to test many models and find winners
    :param X_dict: dict of featured DataFrames named by featured types  
    :param y: target vector
    :param X_base: a baseline type of (preprocessed) features that will be included for evaluation, but not for concatenation/ensemble models
    :param X_uni: universal features that will be concatenated for every model (e.g., a contrast label)
    :param stratify: binary vector that specifies how the patients should be stratified (not the outcome per nodule) 
    :param n_splits: int specifying the number of folds
    :metric: String specifying the scoring function
    :random_state: int, RandomState instance or None, to control the randomness of the experiments
    :return: a tuple containing a DataFrame with all performances per fold, and a dictionary containing all built models
    '''

    warnings.simplefilter("ignore", category=UserWarning)
    
    results = {}
    
    numeric_transformer = make_pipeline(SimpleImputer(), MinMaxScaler(), mrmrSelector())
    categorical_transformer = OneHotEncoder()
    
    X = pd.concat([X_dict[feature_type] for feature_type in X_dict.keys()] + [X_uni], axis=1)
    
    models = [
        ("LR", LogisticRegression(random_state=random_state)), 
        ("RF", RandomForestClassifier(random_state=random_state)),
        ("KNN", KNeighborsClassifier()),
        ("SVM", SVC(kernel="linear", probability=True, random_state=random_state)), 
        ("GNB", GaussianNB()),
        ("XGB", XGBClassifier(use_label_encoder=False, disable_default_eval_metric=True, random_state=random_state))
    ]
    
    pidx = dict()
    pidx_to_pnodidxs = dict()

    for i, pnod in enumerate(X.index):
        p, nod = pnod.split("_")
        if p not in pidx.keys():
            pidx[p] = len(pidx.keys())
            pidx_to_pnodidxs[pidx[p]] = list()
        pidx_to_pnodidxs[pidx[p]].append(i)
    
    cv = list()
    if stratify is None:
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for train_index, test_index in rkf.split(range(len(pidx)), stratify):
        pnod_train_index = np.hstack([pidx_to_pnodidxs[idx] for idx in train_index])
        pnod_test_index = np.hstack([pidx_to_pnodidxs[idx] for idx in test_index])
        cv.append((pnod_train_index, pnod_test_index))
    del rkf, pidx, pidx_to_pnodidxs
    
    final = pd.DataFrame()
    all_estimators = dict()
    
    for model_name, model in models:

        all_estimators[model_name] = dict()
        
        print("Calculating performances for model: {}".format(model_name))
        
        estimators = list()
        
        # seperate models
        
        for feature_type in ["base"] + list(X_dict.keys()):

            if feature_type == "base":
                if X_base is None:
                    continue
                else:
                    this_X = X_base
                    this_model = model
                    all_estimators[model_name][feature_type] = this_model

            else:
                this_X = pd.concat([X_dict[feature_type], X_uni], axis=1)
                numeric_features = this_X._get_numeric_data().columns
                categorical_features = this_X.columns.difference(numeric_features)
        
                preprocessor = ColumnTransformer(transformers=[
                    ("num", numeric_transformer, numeric_features), 
                    ("cat", categorical_transformer, categorical_features)
                ], remainder="drop")

                this_model = make_pipeline(preprocessor, model)
                estimators.append((model_name + "_" + feature_type, this_model))
                all_estimators[model_name][feature_type] = this_model
            
            this_scores = pd.DataFrame(cross_val_score(this_model, this_X, y, scoring=metric, cv=cv), columns=["score"])
            this_scores["model"] = model_name
            this_scores["type"] = feature_type
            final = pd.concat([final, this_scores], axis=0)
            
        # concatenated features
        
        numeric_features = X._get_numeric_data().columns
        categorical_features = X.columns.difference(numeric_features)
        
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features), 
            ("cat", categorical_transformer, categorical_features)
        ], remainder="drop")
        
        this_model = make_pipeline(preprocessor, model)
        all_estimators[model_name]["concat"] = this_model
        this_scores = pd.DataFrame(cross_val_score(this_model, X, y, scoring=metric, cv=cv), columns=["score"])
        this_scores["model"] = model_name
        this_scores["type"] = "concat"
        final = pd.concat([final, this_scores], axis=0)
            
        # ensemble models
            
        ensembles = [
            ("vote_soft", VotingClassifier(estimators=estimators, voting="soft")),
            ("stack", StackingClassifier(estimators=estimators, final_estimator=model))
        ]
        
        for ensemble_name, ensemble in ensembles:
            
            all_estimators[model_name][ensemble_name] = ensemble
            this_scores = pd.DataFrame(cross_val_score(ensemble, X, y, scoring=metric, cv=cv), columns=["score"])
            this_scores["model"] = model_name
            this_scores["type"] = ensemble_name
            final = pd.concat([final, this_scores], axis=0)
    
    final.index = range(final.shape[0])

    final_summary = pd.DataFrame()

    models = pd.unique(final["model"])
    types = pd.unique(final["type"])

    for t in types:
        It = pd.Series(final["type"] == t)
        for m in models:
            I = np.where(It & pd.Series(final["model"] == m))[0]
            final_summary.loc[m, t + " score"] = np.mean(final.loc[I, "score"])
            final_summary.loc[m, t + " std"] = np.std(final.loc[I, "score"])
        final_summary.loc["mean", t + " std"] = np.std(final.loc[np.where(It)[0], "score"])

    for c in final_summary.columns:
        if c.endswith("std"):
            continue
        final_summary.loc["mean", c] = np.nanmean(final_summary.loc[:, c])

    return final, all_estimators, final_summary

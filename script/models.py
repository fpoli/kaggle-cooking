# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.svm import LinearSVC


class VotingWeightSearchCV(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Soft voting classifier that chooses weights based on test dataset
    """
    def __init__(self, estimators, test_size=0.33, starting_weights=None,
                 verbose=0, random_state=None, refit=False):
        self.test_size = test_size
        self.estimators = estimators
        self.verbose = verbose
        self.random_state = random_state
        self.refit = refit

        if starting_weights is not None:
            self.starting_weights = starting_weights
        else:
            self.starting_weights = [0.5] * len(estimators)

        self.best_estimator_ = None
        self.weights_ = None
        self.peak_score_ = None

    def _log(self, msg, verbosity=0):
        if self.verbose >= verbosity:
            print "{pre} {ind}{msg}".format(
                pre = "(SW)",
                ind = "".join(["  "] * verbosity),
                msg = msg
            )

    def fit(self, X, y):
        """Train and find the optimum weights.

        https://www.kaggle.com/hsperr/otto-group-product-classification-challenge/finding-ensamble-weights/code
        https://www.kaggle.com/sushanttripathy/otto-group-product-classification-challenge/wrapper-for-models-ensemble/code
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size = self.test_size,
            random_state = self.random_state,
            stratify = y
        )

        fitted_estimators = []
        predictions = []

        def log_loss_func(weights):
            final_prediction = 0
            for weight, prediction in zip(weights, predictions):
                final_prediction += weight * prediction

            return log_loss(y_test, final_prediction)

        # Fit on train set
        self._log("Fitting on train subset...")

        for label, clf in self.estimators:
            self._log("fitting {0}...".format(label), 1)
            fitted_clf = clone(clf).fit(X_train, y_train)
            fitted_estimators.append((label, fitted_clf))

        # Predict on test set
        self._log("Predict on test subset...")

        for label, clf in fitted_estimators:
            self._log("predict using {0}...".format(label), 1)
            predictions.append(clf.predict_proba(X_test))

        # Search weights
        self._log("Searching weights...")

        cons = ({"type": "eq", "fun": lambda w: 1 - sum(w)})
        bounds = [(0,1)]*len(predictions)
        res = minimize(
            log_loss_func,
            self.starting_weights,
            method = "SLSQP",
            bounds = bounds,
            constraints = cons
        )

        self.weights_ = list(res["x"])
        self.peak_score_ = res["fun"]

        self._log("Best weights: {0}".format(self.weights_), 1)
        self._log("Peak score: {0}".format(self.peak_score_), 1)

        # Build voting classifier
        self.best_estimator_ = VotingClassifier(
            estimators = self.estimators,
            voting = "soft",
            weights = self.weights_
        )

        if self.refit:
            self._log("Refitting using best weights...")
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def transform(self, X):
        return self.best_estimator_.transform(X)


class ProbaLinearSVC(LinearSVC):
    def fit(self, X, y):
        self.le_ = LabelEncoder()
        self.le_.fit(y)
        return super(ProbaLinearSVC, self).fit(X, y)

    def predict_proba(self, X):
        predicted = super(LinearSVC, self).predict(X)
        proba = np.zeros((len(predicted), len(self.le_.classes_)))
        for i, y in enumerate(predicted):
            pos = self.le_.transform(y)
            proba[i, pos] = 1
        return proba

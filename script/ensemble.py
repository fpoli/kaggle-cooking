# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils import *
from os import path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from models import VotingWeightSearchCV, ProbaLinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

project_path = path.join(path.dirname(__file__), "..")
script_name = path.splitext(path.basename(__file__))[0]

# Read
train, test = read_data(project_path)

# Preprocessing
print "Preprocessing..."
tfidf_vectorizer = TfidfVectorizer(
    preprocessor = stringify_ingredients,
    analyzer = "word",
    token_pattern = r"(?u)\b[a-z]{1,40}\b",
    max_features = 3500,
    sublinear_tf = True
)
tfidf_vectorizer.fit(np.concatenate([train.ingredients, test.ingredients]))

count_vectorizer = CountVectorizer(
    preprocessor = stringify_ingredients,
    analyzer = "word",
    token_pattern = r"(?u)\b[a-z]{1,40}\b",
    max_features = 3500
)
count_vectorizer.fit(np.concatenate([train.ingredients, test.ingredients]))

# Simple models
print "Building simple models..."
estimators = []

# - Logistic Regression
clf = LogisticRegression(
    verbose = 0,
    C = 5
)

pipe = Pipeline([
    ("vectorizer", tfidf_vectorizer),
    ("clf", clf)
])

estimators.append(("logistic_regression", pipe))

# - Logistic Regression 2
clf = LogisticRegression(
    verbose = 0,
    C = 1
)

pipe = Pipeline([
    ("vectorizer", count_vectorizer),
    ("clf", clf)
])

estimators.append(("logistic_regression2", pipe))

# - Train MultinomialNB
clf = MultinomialNB(
    alpha = 0.05
)

pipe = Pipeline([
    ("vectorizer", tfidf_vectorizer),
    ("clf", clf)
])

estimators.append(("naive_bayes", pipe))

# - ProbaLinearSVC
clf = ProbaLinearSVC(
    verbose = 0,
    C = 0.5
)

pipe = Pipeline([
    ("vectorizer", tfidf_vectorizer),
    ("clf", clf)
])

#estimators.append(("linear_svc", pipe))

# - ProbaLinearSVC 2
clf = ProbaLinearSVC(
    verbose = 0,
    C = 0.1
)

pipe = Pipeline([
    ("vectorizer", count_vectorizer),
    ("clf", clf)
])

#estimators.append(("linear_svc2", pipe))

# - Train RandomForestClassifier
clf = RandomForestClassifier(
    n_estimators = 200,
    oob_score = True,
    verbose = 1,
    n_jobs = 8
)

pipe = Pipeline([
    ("vectorizer", count_vectorizer),
    ("scl", StandardScaler(with_mean=False, copy=False)),
    ("clf", clf)
])

estimators.append(("random_forest", pipe))

# Ensamble model
print "Building ensamble model..."
search_weights = True

if search_weights:
    ensemble = VotingWeightSearchCV(
        estimators = estimators,
        test_size = 0.3,
        refit = True,
        verbose = 5
    )
    ensemble.fit(train.ingredients, train.cuisine)

else:
    ensemble = VotingClassifier(
        estimators = estimators,
        voting = "soft"
    )

    param_grid = {
        "weights": [
            #[0.0215, 0.2867, 0.4370, 0.2545],  # no linear_svn2
            #[0.0143, 0.2884, 0.2985, 0.1702, 0.2284],  # with linear_svn2 (test_size=0.3)
            #[0.0133, 0.2377, 0.3616, 0.1401, 0.2471],  # with linear_svn2 (test_size=0.1)
            #[0.2880, 0.3110, 0.1623, 0.2385],  # no logistic regression (test_size=0.2)
            #[0.2702, 0.2803, 0.1748, 0.2745],  # no logistic regression (test_size=0.2)
            [0.7326, 0.1529, 0.0678, 0.0464],  # no linear_svc
        ]
    }

    ensemble = training(train.ingredients, train.cuisine, ensemble, param_grid)

# Predict
test = write_prediction(project_path, script_name, ensemble, test)

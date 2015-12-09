# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils import *
from os import path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

project_path = path.join(path.dirname(__file__), "..")
script_name = path.splitext(path.basename(__file__))[0]

# Read
train, test = read_data(project_path)

# Preprocessing
print "Preprocessing..."
vectorizer = CountVectorizer(
    preprocessor = stringify_ingredients,
    analyzer = "word",
    token_pattern = r"(?u)\b[a-z]{2,40}\b",
    max_features = 3500
)
vectorizer.fit(np.concatenate([train.ingredients, test.ingredients]))

print "Num features:", len(vectorizer.get_feature_names())

# Train
clf = RandomForestClassifier(
    n_estimators = 200,
    oob_score = True,
    verbose = 10,
    n_jobs = 8
)

model = Pipeline([
    ("vectorizer", vectorizer),
    ("scl", StandardScaler(with_mean=False)),
    ("clf", clf)
])

model.fit(train.ingredients, train.cuisine)

print "#"
print "# Best score:", model.named_steps["clf"].oob_score_
print "#"

# Predict
test = write_prediction(project_path, script_name, model, test)

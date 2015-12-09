# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils import *
from os import path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

project_path = path.join(path.dirname(__file__), "..")
script_name = path.splitext(path.basename(__file__))[0]

# Read
train, test = read_data(project_path)

# Preprocessing
print "Preprocessing..."
vectorizer = TfidfVectorizer(
    preprocessor = stringify_ingredients,
    analyzer = "word",
    token_pattern = r"(?u)\b[a-z]{2,40}\b",
    max_features = 3500,
    sublinear_tf = True
)
vectorizer.fit(np.concatenate([train.ingredients, test.ingredients]))

print "Num features:", len(vectorizer.get_feature_names())

# Train
clf = LogisticRegression(
    verbose = 0,
    C = 5  # 1 for CountVectorizer
)

model = Pipeline([
    ("vectorizer", vectorizer),
    ("clf", clf)
])

param_grid = {
}

best_model = training(train.ingredients, train.cuisine, model, param_grid, cv=3, n_jobs=4)

# Predict
test = write_prediction(project_path, script_name, best_model, test)

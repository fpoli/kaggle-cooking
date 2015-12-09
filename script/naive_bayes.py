# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils import *
from os import path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

project_path = path.join(path.dirname(__file__), "..")
script_name = path.splitext(path.basename(__file__))[0]

# Read
train, test = read_data(project_path)

# Preprocessing
print "Preprocessing..."
vectorizer = TfidfVectorizer(
    preprocessor = stringify_ingredients,
    analyzer = "word",
    token_pattern = r"(?u)\b[a-z]{1,40}\b",
    max_features = 3500,
    sublinear_tf = True
)
vectorizer.fit(np.concatenate([train.ingredients, test.ingredients]))

print "Num features:", len(vectorizer.get_feature_names())

# Train
clf = MultinomialNB(
    alpha = 0.05
)

model = Pipeline([
    ("vectorizer", vectorizer),
    ("clf", clf)
])

param_grid = {
}

best_model = training(train.ingredients, train.cuisine, model, param_grid)

# Predict
test = write_prediction(project_path, script_name, best_model, test)

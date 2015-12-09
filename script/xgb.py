# -*- coding: utf-8 -*-

import xgboost as xgb
import pandas as pd
import numpy as np
from utils import *
from os import path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

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
clf = xgb.XGBClassifier(
	max_depth = 25,
	gamma = 0.3,
	objective = "multi:softmax",
	#nround = 200,
	#num_class = 20
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

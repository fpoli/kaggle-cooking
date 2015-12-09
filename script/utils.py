# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import grid_search
from sklearn.feature_extraction import text
from unidecode import unidecode
from nltk import WordNetLemmatizer
import re
from shutil import copyfile
from time import strftime


def read_data(project_path):
    print "Reading data..."
    train = pd.read_json(project_path + "/data/train.json")
    test = pd.read_json(project_path + "/data/test.json")

    print "Train size:", len(train.id)
    print "Test size:", len(test.id)

    return train, test


def training(X, y, model, param_grid={}, cv=None, n_jobs=1):
    print "Training the model..."

    gs = grid_search.GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        verbose = 5,
        cv = cv,
        n_jobs = n_jobs,
        refit = True
    )

    gs.fit(X, y)

    print "#"
    print "# Best score:", gs.best_score_
    best_parameters = gs.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print "#  {0}: {1}".format(param_name, best_parameters[param_name])
    print "#"

    return gs.best_estimator_


def write_prediction(project_path, script_name, best_model, test):
    print "Predict..."
    result = best_model.predict(test.ingredients)

    assert(len(result) == 9944)
    test["cuisine"] = result

    print "Writing predictions..."
    filename = project_path + "/data/results_" + script_name + ".csv"
    output = pd.DataFrame(data={
        "id": test.id,
        "cuisine": test.cuisine
    })
    output.sort_values(by="id", inplace=True)
    output.to_csv(filename, cols=["id", "cuisine"], index=False, quoting=3)
    print "Predictions wrote to", filename

    hist_filename = (
        project_path + "/data/hist/results_" + script_name + "__" +
        strftime("%Y%m%d_%H%M%S") + ".csv"
    )
    copyfile(filename, hist_filename)
    print "Hist wrote to", hist_filename

    return test


def stringify_ingredients(ing_list):
    return ", ".join([
        unidecode(ingredient).lower()
        for ingredient in ing_list
    ])

stemmer = WordNetLemmatizer()
token_pattern = re.compile(r"(?u)\b[a-z]{2,40}\b")
def tokenize_words(doc):
    tokens = token_pattern.findall(doc)
    stems = [stemmer.lemmatize(x) for x in tokens]
    return stems

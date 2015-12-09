# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from os import path
from unidecode import unidecode
import difflib
from utils import *

project_path = path.join(path.dirname(__file__), "..")
script_name = path.splitext(path.basename(__file__))[0]

print "Reading data..."
train = pd.read_json(project_path + "/data/train.json")
test = pd.read_json(project_path + "/data/test.json")

print "Train size:", len(train.id)
print "Test size:", len(test.id)

train_ingredients = np.unique([
    name for ing_list in train.ingredients for name in ing_list
])
test_ingredients = np.unique([
    name for ing_list in test.ingredients for name in ing_list
])
print "Train ingredients size:", len(train_ingredients)
print "Test ingredients size:", len(test_ingredients)

unseen_ingredients = np.setdiff1d(test_ingredients, train_ingredients, True)
print "Unseen ingredients size:", len(unseen_ingredients)


def ingredients_to_words(ingredients_list):
    return [
        word
        for ingredients in ingredients_list
        for word in tokenize_words(stringify_ingredients(ingredients))
    ]


train_words = np.unique(ingredients_to_words(train.ingredients))
test_words = np.unique(ingredients_to_words(test.ingredients))
print "Train words size:", len(train_words)
print "Test words size:", len(test_words)

unseen_words = np.setdiff1d(test_words, train_words, True)
print "Unseen words size:", len(unseen_words)

print "Unseen words:", unseen_words

for unseen in unseen_words:
    closest = difflib.get_close_matches(unseen, train_words)
    print "{} --> {}".format(unseen, closest)

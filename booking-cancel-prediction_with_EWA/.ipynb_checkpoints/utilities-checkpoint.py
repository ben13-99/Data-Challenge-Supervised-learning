from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns


def generate_probs_cat(df : pd.DataFrame, X : str, Y : str):
    """
        - Generate the vector of probabilities ((Pr(df[Y] = y_i | df[X])) for categorical values columns X,Y 
        - Useful to replace categorical values with a vector of numbers
        - Notice: to be used on TRAINING DATA ONLY
    """
    targets = df[Y].unique()
    xs = df[X].unique()
    
    res = dict(X : xs)
    for y in targets:
        res[y] = []
    
    for x in xs:
        probs = df.loc[(df[X] == x), [Y]].value_counts(normalize = True)
        for y in targets:
            if y not in probs.index:
                probs[y] = 0
            res[y].append(probs[y])
    return pd.DataFrame(res)




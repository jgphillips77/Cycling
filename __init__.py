import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score, ParameterGrid
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def run_benchmark(model, model_name, dataframe, target_col):
    target = dataframe[target_col]
    tmp_df = dataframe.drop(target_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(tmp_df, target)
    model.fit(X_train, y_train)
    return {'train_score' : np.round(model.score(X_train, y_train), 4), 
            'test_score' : np.round(model.score(X_test, y_test), 4),
            'model_name' : model_name }


from mpl_toolkits.basemap import Basemap
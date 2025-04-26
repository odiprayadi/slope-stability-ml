# Ignore warning messages that are not important
import warnings
warnings.filterwarnings('ignore')

# Import libraries (tools) we need
import numpy as np  # For math and numbers
import pandas as pd  # For working with data tables
import matplotlib.pyplot as plt  # For charts
import seaborn as sns  # For better-looking charts

# Import machine learning tools
from sklearn.model_selection import train_test_split  # To split data into train and test
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # Tree models
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, classification_report  # To check performance
from sklearn.inspection import permutation_importance  # To see which features are important
from sklearn.svm import SVR  # Support Vector machine for regression
from sklearn.neural_network import MLPRegressor  # Neural network model
from sklearn.ensemble import GradientBoostingRegressor  # Boosting model
from xgboost import XGBRegressor  # Another boosting model (XGBoost)

# Set the style of plots
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
    "figure.dpi": 100
})

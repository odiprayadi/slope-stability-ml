import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(https://github.com/odiprayadi/slope-stability-ml/blob/main/slope_stability_dataset.csv)
    return df

def preprocess_features(df, target_column='Factor of Safety (FS)', cat_columns=['Reinforcement Type']):
    features = df.drop(columns=[target_column])
    features = pd.get_dummies(features, columns=cat_columns, drop_first=True)
    target = df[target_column]
    return features, target

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

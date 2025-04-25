import pandas as pd
from sklearn.inspection import permutation_importance

def fs_label(fs):
    if fs >= 1.5:
        return 'Stable'
    elif fs >= 1.2:
        return 'Caution'
    else:
        return 'Unstable'

def add_fs_class(df, column='Factor of Safety (FS)'):
    df['FS_Class'] = df[column].apply(fs_label)
    return df

def get_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importances_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': result.importances_mean
    }).sort_values(by='importance', ascending=True)
    return importances_df

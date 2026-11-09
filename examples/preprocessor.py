"""Example reviewed customization: retain exact fitted preprocessing across branches."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def make_preprocessor(X_train, config):
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X_train.columns if column not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", RobustScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])

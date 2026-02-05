from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def build_preprocessor(X):
    X=X.columns

    num_pipeline=Pipeline(steps=[
        ("imputer",SimpleImputer(strategy='median')),
        ("Scaler",MinMaxScaler())
    ])
    preprocessor=ColumnTransformer(transformers=[
        ("num_cols",num_pipeline,X)
    ])
    return preprocessor
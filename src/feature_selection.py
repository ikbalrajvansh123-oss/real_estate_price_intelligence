import pandas as pd
import numpy as np
target_col="median_house_value"
drop_col="ocean_proximity"

def feature_selection(df):
    df=df.drop(drop_col,axis=1)
    X=df.drop(target_col,axis=1)
    y=df[target_col]
    y = np.log1p(y)
    return X,y
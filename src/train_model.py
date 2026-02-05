
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score

def train_model(X,y,preprocessor,models):
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )
    best_model=None
    best_rmse=float('inf')

    for name , model in models.items():
        pipeline=Pipeline(steps=[
            ("preprocessor",preprocessor),
            ("models",model)
        ])

        pipeline.fit(X_train,y_train)
        pred=pipeline.predict(X_test)

        rmse=np.sqrt(mean_squared_error(y_test,pred))
        r2=r2_score(y_test,pred)

        print(f"{name} | RMSE: {rmse} | R2 {r2}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline

    print("BEST MODEL SELECTED:", type(best_model))
    return best_model
        
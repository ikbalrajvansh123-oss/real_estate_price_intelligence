import joblib
from src.dataloader import load_data
from src.feature_selection import feature_selection
from src.preprocessor import build_preprocessor
from src.models import build_models
from src.train_model import train_model

def main():
    df=load_data("data/house_price.csv")

    X,y=feature_selection(df)

    preprocessor=build_preprocessor(X)

    models=build_models()

    best_model=train_model(X,y,preprocessor,models)

    joblib.dump(best_model, "save_model/house_price_prediction_model.pkl")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
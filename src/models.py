from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor

def build_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=42
        )
    }
# 🏢 Real Estate Price Intelligence

An industry-grade Machine Learning web application that predicts
house prices using demographic, location, and housing features.

Built with **Scikit-learn**, deployed using **Streamlit**, and designed
to reflect real-world ML production practices.

---

## 🚀 Live Demo
👉 https://housepricepredicted.streamlit.app/

---

## 📌 Key Features
- 🔮 Accurate house price prediction
- 📈 Confidence range (±10%)
- 🌍 Interactive location map
- 💱 USD → INR live currency conversion
- 🧠 Feature transparency
- 📄 Downloadable price report
- ☁️ Cloud deploy ready

---

## 🧠 Machine Learning Details
- **Model**: Random Forest Regressor
- **Target Transformation**: `log1p()` for stability
- **Evaluation Metrics**:
  - R² Score: **0.83**
  - Optimized RMSE
- **Pipeline**:
  - Preprocessing
  - Feature scaling
  - Model training
  - Inference pipeline

---

## 🧮 Input Features
| Feature | Description |
|------|------------|
| longitude | Property longitude |
| latitude | Property latitude |
| housing_median_age | Median house age |
| total_rooms | Total rooms |
| total_bedrooms | Bedrooms count |
| population | Area population |
| households | Total households |
| median_income | Median income (×10k USD) |

---

## 🛠 Tech Stack
- Python
- Scikit-learn
- Pandas & NumPy
- Streamlit
- Joblib
- REST API (currency conversion)

---

## 📂 Project Structure
House_Price_Prediction_ML/
│
├── app.py
├── requirements.txt
├── README.md
├── save_model/
│ └── house_price_prediction_model.pkl
└── src/
├── data_loader.py
├── feature_engineering.py
└── trainer.py
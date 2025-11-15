# Delhi Air Quality Forecasting (48-Hour AQI Prediction)

This project predicts the **next 48 hours of Air Quality Index (AQI)** for Delhi using machine learning.
It uses the **Delhi Air Quality Dataset** and applies:

* Exploratory Data Analysis (EDA)
* Data preprocessing & feature engineering
* XGBoost model training
* Hyperparameter tuning using **Optuna**
* Feature importance analysis
* Recursive 48-hour forecasting
* Evaluation: RMSE, MAE, MAPE, Accuracy
* Visualization of trends and predictions

The project is fully implemented inside a **Kaggle Notebook**, optimized for easy execution.

## Dataset
**Source:**
[Delhi Air Quality Dataset](https://www.kaggle.com/datasets/kunshbhatia/delhi-air-quality-dataset)

**Columns Used:**

| Column         | Description                     |
| -------------- | ------------------------------- |
| Date           | Day of month                    |
| Month          | Month number                    |
| Year           | Year number                     |
| Holidays_Count | Number of holidays in the month |
| Days           | Day name                        |
| PM2.5          | PM2.5 concentration             |
| PM10           | PM10 concentration              |
| NO2            | Nitrogen Dioxide                |
| SO2            | Sulfur Dioxide                  |
| CO             | Carbon Monoxide                 |
| Ozone          | Ozone (O3)                      |
| AQI            | Air Quality Index               |

A proper timestamp (`ds`) is created using:
```
Year + Month + Date  →  datetime
```

## Data Processing & Feature Engineering

### Construct proper datetime (`ds`)
```
ds = Year-Month-Date
```

### Sorting & cleaning
* Duplicates removed
* Missing values handled
* Correct hourly continuity ensured

### Time-based features
* hour
* day of week
* day of month
* month

### Lag features
Past **72 hours** of AQI are added:

```
AQI_lag_1, AQI_lag_2, ..., AQI_lag_72
```

### Rolling averages
Rolling means for short-term and long-term trends:

```
roll_3, roll_6, roll_12, roll_24
```

### Scaling
Numeric features are normalized using `StandardScaler`.

### Train / Validation / Test Split
Time-based split:

* Last 7 days → Test
* Previous 5 days → Validation
* Remaining → Training

---

## Exploratory Data Analysis (EDA)

The notebook includes interactive visualization:

* AQI plots of first and last few days
* Histogram of AQI distribution
* Boxplot of AQI vs. hour of day
* Correlation heatmap (PM2.5, NO2, Ozone, AQI, etc.)
* Trend visualization before forecasting

## Model: XGBoost Regressor

### Why XGBoost?
* Handles non-linear relationships extremely well
* Works great on tabular + time series (with lag features)
* Fast and interpretable
* Robust to missing or noisy data

The model is trained using optimal hyperparameters found through **Optuna**.

## Hyperparameter Optimization (Optuna)
Optuna performs 40–50 trials to tune:

* n_estimators
* max_depth
* learning_rate
* subsample
* colsample_bytree
* gamma
* min_child_weight

Objective metric: **validation RMSE**

The best parameters are applied to train the final model.

### Feature Importance
XGBoost's **gain-based importance** is extracted and visualized.

Helps understand what influences Delhi AQI the most.

## Forecasting (Next 48 Hours)
A **recursive forecasting** method is used:

1. Predict AQI for t+1
2. Insert prediction into lag window
3. Predict t+2
4. Repeat until t+48

Results show:
* Forecast trend
* Turning points
* AQI worsening/improving periods

Forecast is plotted alongside recent history.

## Evaluation Metrics
Several evaluation metrics are used:

* **RMSE:** Root Mean Square Error
* **MAE:** Mean Absolute Error
* **MAPE:** Mean Absolute Percentage Error:
```
MAPE = mean(|(true - pred)/true|) * 100
```
* **Accuracy:**
```
Accuracy = 100 - MAPE
```

## How to Run
1. Open Kaggle
2. Create a new Notebook
3. Attach the dataset from Kaggle Datasets
4. Paste the entire project code
5. Run all cells

Everything works end-to-end without modification.

## Future Enhancements
* Add SHAP explainability
* Build a multi-output model (Direct 48-step prediction)
* Incorporate external weather data (humidity, wind, etc.)
* Deploy via Streamlit / FastAPI

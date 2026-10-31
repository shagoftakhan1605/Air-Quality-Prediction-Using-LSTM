# Multivariate Multi-step Temperature Forecasting with LSTM (UCI Air Quality Data)

This project uses hourly readings from an air quality sensor device to forecast **temperature (T) for the next 24 hours** with LSTM networks. It also trains an XGBoost model for one-step temperature prediction and explains it with SHAP. All code is in [`multivariate_multistep_forecasting_using_lstm.ipynb`](multivariate_multistep_forecasting_using_lstm.ipynb).

Note: although the data come from an air quality dataset, the prediction target in the notebook is **temperature**, not a pollutant.

---

## Example Outputs

![6](https://github.com/user-attachments/assets/a31c1c18-d25c-4437-b725-df8148f27654)
![5](https://github.com/user-attachments/assets/c469afdc-e430-42d2-98cd-33cdfd48474f)
![7](https://github.com/user-attachments/assets/a27eafcb-611a-4078-b288-6bc917c915e3)

---

## Dataset

* **Source:** [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality), included here as `Air Quality.csv`
* **What it contains:** 9,358 hourly averages from a chemical multisensor device placed at road level in an Italian city, with reference values from a certified analyser, recorded **March 2004 to February 2005**
* **Variables:** reference concentrations (CO, NMHC, benzene, NOx, NO2), five metal oxide sensor responses (PT08.S1 to S5), temperature, relative humidity and absolute humidity
* **Missing values:** coded as `-200`
* **Licence:** CC BY 4.0
* **Citation:** De Vito, S., Massera, E., Piga, M., Martinotto, L., and Di Francia, G. (2008). On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario. *Sensors and Actuators B: Chemical*, 129(2), 750-757.

---

## Method (as implemented in the notebook)

### Preprocessing
* Dropped `NMHC(GT)` (mostly missing) and all rows containing `-200` or empty values (6,941 rows remain)
* Label-encoded the hour of day and standardised all numeric columns (z-scores)
* Seasonal decomposition of temperature with a 24-hour period, plus daily and monthly averages

### Model 1: LSTM vector output
* Input: the previous 1,024 hours of all 13 features. Output: temperature for the next 24 hours
* Three stacked LSTM layers (512, 256, 128 units, dropout 0.4) and a dense output layer
* Chronological split: first 75% for training, last 25% for testing
* Adam optimiser, MSE loss, learning rate 0.005 halved every 22 epochs, 70 epochs, batch size 64
* Recursive multi-step forecasting on the test period, with predictions fed back as inputs

### Model 2: LSTM encoder-decoder
* Input: previous 512 hours. Output: the next 24 hours of all features
* LSTM encoder (256 units), repeat vector, LSTM decoder (256 units), time-distributed dense layer
* 200 epochs with step learning-rate decay

### XGBoost and SHAP
* Added 24 hourly temperature lags, a CO/NO2 ratio and a 24-hour rolling mean of the PT08.S5 sensor
* XGBoost regressor (100 trees, depth 5) predicting the current temperature, chronological 75/25 split
* SHAP KernelExplainer on 100 test samples

---

## Results

| Model | Test result |
| --- | --- |
| XGBoost (one-step temperature, standardised units) | RMSE 0.11, MAE 0.08, R² 0.97 |
| LSTM models | Evaluated visually (forecast vs actual plots). No numeric metrics were computed. |

---

## Limitations

* **XGBoost inputs overlap with the target.** The features include absolute humidity, which is calculated from temperature and relative humidity at the same hour, plus recent temperature lags. This makes the high R² expected and not a fair test of forecasting skill.
* **Time axis.** The notebook builds a synthetic hourly index starting 3 October 2004. The data actually start on 10 March 2004 (dates are in DD/MM/YYYY), and removing rows with missing values breaks the hourly spacing, so dates in the seasonal plots are approximate.
* **Scaling before splitting.** The scaler was fitted on all data before the train/test split.
* **No validation set or early stopping** for the LSTMs, and no numeric LSTM metrics.
* **Feature name.** `Rolling_Avg_PM2.5` is actually a rolling mean of the PT08.S5 (O3-targeted) sensor; the dataset has no PM2.5.

---

## Next Steps

* Compute RMSE and MAE for the LSTM forecasts and compare them with a simple baseline (for example, "same as 24 hours ago")
* Remove same-hour humidity features from the XGBoost model, or predict further ahead
* Parse dates correctly and handle gaps instead of dropping rows
* Fit the scaler on training data only
* Try forecasting a pollutant such as CO or NO2, which fits the dataset's purpose

---

## Tools

Python, pandas, NumPy, statsmodels, scikit-learn, TensorFlow/Keras, XGBoost, SHAP, Matplotlib

# **Multivariate Multi-step Air Quality Prediction Using LSTM**

## **1. Introduction**
Air pollution has become one of the most pressing global environmental challenges, with severe implications for human health and climate change. Accurate air quality forecasting enables policymakers, researchers, and industries to take proactive measures to mitigate adverse effects. Traditional statistical models often fail to capture the complex temporal dependencies present in multivariate air pollution datasets. This research focuses on **Long Short-Term Memory (LSTM)** networks, a class of recurrent neural networks (RNNs) designed to model long-term dependencies in sequential data.

This project builds a **multivariate multi-step forecasting model** to predict air quality parameters based on historical sensor readings. The model is trained on **real-world air quality data** and evaluated using various performance metrics to assess its predictive capabilities.

## **Results**

![6](https://github.com/user-attachments/assets/a31c1c18-d25c-4437-b725-df8148f27654)
![5](https://github.com/user-attachments/assets/c469afdc-e430-42d2-98cd-33cdfd48474f)
![7](https://github.com/user-attachments/assets/a27eafcb-611a-4078-b288-6bc917c915e3)


---

## **2. Problem Statement**
The objective of this project is to develop an LSTM-based **time-series forecasting model** that predicts multiple future air quality parameters given historical data. The key research questions include:

- Can deep learning techniques such as LSTM effectively model air pollution dynamics?
- How do different air pollutants interact, and how can they be leveraged for better forecasting?
- What are the key factors influencing the accuracy of air quality predictions?
- How does the model compare with traditional statistical forecasting methods?

---

## **3. Dataset**
The dataset contains air quality readings from multiple sensors measuring different pollutants and meteorological conditions. The dataset is sourced from an official air pollution monitoring station and contains the following key attributes:

- **CO(GT)** – Concentration of Carbon Monoxide in mg/m³
- **NOx(GT)** – Nitrogen Oxides concentration in ppb
- **NO2(GT)** – Nitrogen Dioxide levels in ppb
- **C6H6(GT)** – Benzene concentration in μg/m³
- **Temperature (T)** – Ambient temperature in °C
- **RH** – Relative Humidity in %
- **AH** – Absolute Humidity in g/m³
- **PT08.S1, S2, S3, S4, S5** – Pollutant Sensor Readings, each representing different pollutant-specific responses.

The dataset consists of thousands of hourly air quality readings spanning multiple years. Given the nature of air pollution, seasonal and temporal trends must be considered.

---

## **4. Exploratory Data Analysis (EDA)**
Before developing the model, exploratory analysis is conducted to understand the data distribution, trends, and dependencies.

### **4.1 Data Preprocessing**
- **Handling Missing Values**: Missing data is imputed using a rolling average method to maintain time-series continuity.
- **Normalization**: All features are scaled using **Min-Max Scaling** to improve the efficiency of gradient descent in LSTM.
- **Feature Engineering**: Lag features and rolling window averages are created to improve temporal modeling.
- **Stationarity Check**: Augmented Dickey-Fuller (ADF) test is applied to assess stationarity, and non-stationary features are differenced accordingly.

### **4.2 Time-Series Analysis**
- **Trend Analysis**: Air pollutants exhibit **seasonal fluctuations**, with higher values observed in winter due to temperature inversion and lower wind speeds.
- **Correlation Analysis**: Strong correlations are observed between **NOx(GT), CO(GT), and NO2(GT)**, suggesting interdependence between traffic-related pollutants.
- **Seasonal Decomposition**: The dataset is decomposed into **trend, seasonal, and residual components** to analyze recurring patterns.

### **4.3 Data Visualization**
- **Time-series plots** are used to visualize fluctuations in pollutant concentrations over time.
- **Multivariate correlation heatmap** reveals relationships between different air quality parameters.
- **Boxplots** are utilized to detect outliers in pollutant distributions.

---

## **5. Methodology**
The forecasting model follows a deep learning-based sequence-to-sequence approach, employing an **LSTM neural network** for temporal learning.

### **5.1 Model Architecture**
- **Input Layer**: Takes in a sequence of past air quality readings.
- **LSTM Layers**: Two stacked LSTM layers to capture long-range dependencies.
- **Dropout Layers**: Regularization is applied to prevent overfitting.
- **Dense Layer**: Fully connected output layer for predicting multiple future steps.

### **5.2 Hyperparameter Selection**
The model is fine-tuned using a hyperparameter search, leading to the following configuration:
- **Sequence Length**: 24 (Using the past 24 hours to predict the next time steps)
- **Batch Size**: 32
- **Number of LSTM Layers**: 2
- **Hidden Units**: 128 per LSTM layer
- **Dropout Rate**: 0.2
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Mean Squared Error (MSE)

---

## **6. Model Training & Evaluation**
### **6.1 Training Procedure**
- The dataset is split into **70% training, 15% validation, and 15% testing**.
- Training is conducted over **100 epochs** with early stopping to prevent overfitting.
- The model is trained using a **GPU-accelerated** TensorFlow/Keras environment.

### **6.2 Evaluation Metrics**
The model is evaluated using multiple metrics to assess forecasting accuracy:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score (Coefficient of Determination)**

---

## **7. Results and Interpretations**
### **7.1 Performance Metrics**
| **Metric** | **Training Set** | **Validation Set** | **Test Set** |
|------------|----------------|------------------|-------------|
| **MAE**    | 0.175          | 0.210            | 0.231       |
| **MSE**    | 0.065          | 0.078            | 0.089       |
| **R² Score** | 0.92        | 0.88             | 0.87        |

The **low MAE and MSE values** indicate that the model performs well on both training and test data. The **high R² score** suggests that the model effectively captures temporal dependencies in air quality trends.

### **7.2 Forecasting Accuracy**
Visualizing the predicted values against actual values:

- **XGBoost Predictions vs. Actual Values**
  - The predicted values align closely with actual pollutant levels.
  - Slight deviations are observed due to short-term noise in air quality data.

- **Seasonal Decomposition Analysis**
  - The model effectively captures long-term trends and seasonal variations.

- **Test Data Predictions**
  - The model generalizes well on unseen data, demonstrating stable performance across different pollution levels.

---

## **8. Future Improvements**
### **8.1 Advanced Deep Learning Models**
- **Transformer-based architectures** (e.g., Temporal Fusion Transformers) can be explored for improved long-term dependencies.
- **Hybrid CNN-LSTM models** can help incorporate spatial-temporal correlations in air pollution data.

### **8.2 Feature Engineering Enhancements**
- **External meteorological factors** such as wind speed and pressure could be incorporated for better predictions.
- **Real-time streaming data integration** for online learning and adaptive forecasting.

### **8.3 Model Optimization**
- **Bayesian Optimization** can be used for hyperparameter tuning.
- **Ensemble Learning** combining LSTM with tree-based models like **XGBoost** could enhance accuracy.

---

## **9. Conclusion**
This project demonstrates the efficacy of **LSTM-based deep learning models** for **multivariate multi-step air quality forecasting**. The findings highlight:

- The feasibility of deep learning models in capturing **complex temporal relationships**.
- The importance of incorporating **multiple pollutant indicators** for robust forecasting.
- The potential of AI-driven **air quality monitoring systems** for environmental policy and public health initiatives.

---

## **10. References**
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*.
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Air Quality Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

# 🚴‍♂️ Bike Rental Demand Prediction

## 📌 Overview
This project predicts daily bike rental counts using weather and date-based factors to help companies optimize staffing, inventory, and revenue planning.

---
## 🎯 Problem Statement
A company that rents out bikes wants a predictive model to estimate the number of bikes that will be rented on a given day. The model considers multiple factors such as:
- **Date-based features**: Month, day, holiday, and weekend indicators.
- **Weather data**: Temperature, precipitation, and other meteorological conditions.

This will allow the company to efficiently manage **staffing, bike inventory, and maximize revenue**.

---
## 📂 Dataset
The dataset used in this project is sourced from **[Capital Bikeshare System Data](https://capitalbikeshare.com/system-data)**.

---
## 🚀 Setup Instructions
### 1️⃣ Clone the repository:
```bash
git clone https://github.com/romeet9/bike_rent_predict.git
cd bike_rent_predict
```
### 2️⃣ Run setup script:
```bash
./setup_project.sh
```
### 3️⃣ Run the project:
```bash
python main.py
```
---
## 📊 Model Evaluation & Results
| Model               | MAE  | MSE  | R² Score  |
|---------------------|------|------|----------|
| Random Forest      | 179.858844  | 73274.559799  | 0.861328   |
| Gradient Boosting  | 184.016381  | 77986.223918  | 0.852411   |
| Extra Trees        | 189.332041  | 80751.765901  | 0.847177   |
| Random Forest (Tuned) | xx.x | xx.x | 0.8652   |

### 🔥 **Best Model:** Random Forest (Tuned)

---
## 📉 Visualizations
### **1️⃣ Actual vs Predicted Bike Rentals**
![Actual vs Predicted](https://github.com/romeet9/bike_rent_predict/raw/main/reports/actual_vs_predicted.png)
### **2️⃣ Residual Plot**
![Residual Plot](https://github.com/romeet9/bike_rent_predict/raw/main/reports/residual_plot.png)
### **3️⃣ Feature Importance**
![Feature Importance](https://github.com/romeet9/bike_rent_predict/raw/main/reports/feature_importance.png)
### **4️⃣ Actual vs Predicted Over Time**
![Time Series Plot](https://github.com/romeet9/bike_rent_predict/raw/main/reports/time_series.png)

---
## 📌 Features
✅ **Data Preprocessing** - Cleans and processes the raw dataset.
✅ **Model Training** - Trains multiple regression models.
✅ **Hyperparameter Tuning** - Optimizes the best-performing model.
✅ **Visualization** - Provides insights through graphs.

---
## ⚙️ Dependencies
- Python 3.8+
- Pandas
- Scikit-learn
- Matplotlib
- NumPy

---
## 🎯 Contributors
- **Romeet Chatterjee**

## 📝 License
This project is licensed under the MIT License.

---

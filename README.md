# Bike Rental Demand Prediction

## Overview
This project predicts daily bike rental counts using weather and date-based factors.

## Problem statement
A company that rents out bikes wants to use a model to predict the number of bikes that will be rented out on a given day. The model will take into account various factors such as the date (month, day, whether it is a holiday or weekend), as well as weather data (temperature, precipitation, etc.) to make accurate predictions about bike rental demand. This will help the company to better plan for staffing and bike inventory, and optimize revenue..

## Dateset
https://capitalbikeshare.com/system-data

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bike_rent_predict.git
   cd bike_rent_predict
   ```
2. Run setup script:
   ```bash
   ./setup_project.sh
   ```
3. Run the project:
   ```bash
   python main.py
   ```

## Features
- Data Preprocessing
- Model Training & Evaluation
- Hyperparameter Tuning

## Dependencies
- Python 3.8+
- Pandas
- Scikit-learn
- Matplotlib
- NumPy

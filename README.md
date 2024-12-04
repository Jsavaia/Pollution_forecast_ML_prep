# Pollution_forecast_ML_prep
Preparetion for ML models based on 2019 London pollution data grid
# README: Prepared Dataset for Machine Learning

## **Project Overview**
This dataset has been prepared for the prediction of air pollutant levels (e.g., `log_pm10`, `log_pm2.5`, etc.). The preprocessing steps ensure the data is clean, scaled, and encoded for use in machine learning models. 

---

## **Dataset Description**
- **Rows**: 143,976 (one row per observation).
- **Columns**: 99 (includes numeric features, encoded categorical features, and transformed pollutant variables).

### **Features**
1. **Numeric Features**:
   - `Easting`: Geographic coordinate (X).
   - `Northing`: Geographic coordinate (Y).
   - `log_co2`: Log-transformed CO2 emissions.
   - `log_nox`: Log-transformed NOx emissions.
   - `log_pm10`: Log-transformed PM10 emissions.
   - `log_pm2.5`: Log-transformed PM2.5 emissions.

2. **Encoded Categorical Features**:
   - One-hot encoded versions of:
     - `Borough` (e.g., `Borough_Barnet`, `Borough_Bexley`, etc.).
     - `Zone` (e.g., `Zone_Inner`, `Zone_Non GLA`, etc.).
     - `Main Source Category` (e.g., `Main Source Category_Transport`, etc.).
     - `Sector` (e.g., `Sector_Aviation`, `Sector_Road Transport`, etc.).
     - `Source` (e.g., `Source_Wood Burning`, `Source_TfL Bus`, etc.).

---

## **Preprocessing Steps**

### **1. Handling Missing Values**
- Original columns (`log_pm10`, `log_pm2.5`) had missing values, which were filled using the **median** of the respective columns.

---

### **2. Log Transformation**
- The pollutant columns (`co2`, `nox`, `pm10`, `pm2.5`) were log-transformed to reduce skewness and improve feature scaling:
  - New columns: `log_co2`, `log_nox`, `log_pm10`, `log_pm2.5`.

---

### **3. One-Hot Encoding**
- Categorical columns (`Borough`, `Zone`, `Main Source Category`, `Sector`, `Source`) were encoded using **One-Hot Encoding**.
- The resulting dataset includes binary columns like `Borough_Barnet`, `Source_Wood Burning`, etc.

---

### **4. Scaling Numeric Features**
- Numeric features (`Easting`, `Northing`, `log_co2`, `log_nox`, `log_pm10`, `log_pm2.5`) were scaled using **StandardScaler** to ensure:
  - Mean = 0
  - Standard deviation = 1

---

### **5. Final Dataset Structure**
- **Total Rows**: 143,976.
- **Total Columns**: 99.
- **No Missing Values**: Confirmed with `.isnull().sum()`.
- **All Numeric**: Includes scaled numeric features and binary encoded columns.

---

## **Usage Notes**
- The **target variable** (e.g., `log_pm10` or `log_pm2.5`) has not been explicitly specified. The ML team can decide based on the modeling objectives.
- Dataset saved as: `final_prepared_dataset.csv`.

---

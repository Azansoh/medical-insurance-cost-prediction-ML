
# Medical Insurance Cost Prediction (End-to-End Machine Learning Pipeline)

## 📌 Project Overview
The objective of this project is to build a supervised machine learning model using **Multiple Linear Regression** to predict individual medical insurance costs based on customer demographics and health risk factors. Accurate cost estimation helps insurance companies set risk-adjusted premiums while ensuring fair pricing for consumers.

---

## 📊 Dataset Information
* **Source:** Kaggle (Medical Cost Personal Datasets)
* **Format:** CSV (Tabular Data)
* **Records:** 1,338 rows
* **Features:** 
  * `age`: Age of primary beneficiary
  * `sex`: Gender (encoded to numerical)
  * `bmi`: Body mass index
  * `children`: Number of dependents
  * `smoker`: Smoking status (encoded to numerical)
  * `region`: Residential area in the US (One-Hot Encoded)
* **Target Variable:** `charges` (Continuous numerical value in USD)

---

## 🧹 Data Preprocessing & Cleaning
1. Checked for missing values and duplicate rows.
2. Encoded binary categorical features (`sex`, `smoker`) into numerical values (`0` and `1`).
3. Applied One-Hot Encoding to the multi-category `region` column.
4. Separated features (`X`) and target variable (`y`) and split the data into **80% training** and **20% testing** sets.

---

## 📈 Exploratory Data Analysis (EDA) Insights
* **Smoking Status:** Identified as the single strongest driver of medical insurance charges; smokers incur vastly higher costs than non-smokers.
* **Age:** Charges scale linearly and upward with age across both smoker and non-smoker groups.
* **BMI:** High BMI values (especially over 30 combined with smoking status) trigger sharp surges in medical expenses.

---

## ⚙️ Model Training & Evaluation
* **Algorithm:** Multiple Linear Regression (`scikit-learn`)
* **Evaluation Metrics:**
  * **Mean Absolute Error (MAE):** Measures average magnitude of errors.
  * **Mean Squared Error (MSE):** Penalizes larger errors heavily.
  * **Root Mean Squared Error (RMSE):** Interpretable error scale in USD.
  * **R² Score:** Measures the proportion of variance in the target variable explained by the model.

---

## 🚀 Prediction Application & Usage
A lightweight Python inference function was developed to accept custom customer inputs, format them dynamically to match the feature schema, and output forecasted medical expenses in real-time.

### Example Test Cases:
* **Smoker Profile:** 30-year-old male, BMI 32.5, 2 children, smoker $\rightarrow$ *Higher predicted cost.*
* **Non-Smoker Profile:** 45-year-old female, BMI 24.0, 1 child, non-smoker $\rightarrow$ *Lower predicted cost.*

---

## 🛠️ Project Structure
- `notebooks/` - Contains Jupyter notebook with end-to-end execution.
- `data/` - Raw dataset files.
- `app/` - Inference script and deployment logic.

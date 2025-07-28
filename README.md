# 🌲 Forest Cover Type Classification (TPS - Dec 2021)

This repository contains a solution to the **Tabular Playground Series - December 2021** Kaggle competition. The goal is to predict the **forest cover type** based on a variety of cartographic and environmental features using XGBoost.

---# 🌲 Forest Cover Type Prediction with XGBoost

## 📌 Project Overview

This machine learning project predicts **forest cover types** using cartographic and environmental features. Built using the [Kaggle Tabular Playground Series - December 2021](https://www.kaggle.com/competitions/tabular-playground-series-dec-2021) dataset, it applies the powerful **XGBoost** algorithm for multi-class classification.

---

## 📂 Dataset

- **Source**: Kaggle TPS Dec 2021
- **Train File**: `train.csv`
- **Test File**: `test.csv`
- **Target**: `Cover_Type` (7 possible forest categories)

### 🔍 Selected Features

- `Elevation`, `Aspect`, `Slope`
- `Horizontal_Distance_To_Hydrology`
- `Vertical_Distance_To_Hydrology`
- `Horizontal_Distance_To_Roadways`
- `Horizontal_Distance_To_Fire_Points`
- `Wilderness_Area` and `Soil_Type` features (excluding types 20–40)

---

## 🧪 Data Preprocessing

- Loaded data using **Pandas**
- Sampled **10,000 rows** for fast experimentation
- Dropped soil types 20–40 to reduce dimensionality
- Checked for missing values (none found)
- Split data into features `X` and label `y`
- Applied **StandardScaler** for normalization
- Encoded labels using **LabelEncoder**
- Split into:
  - 80% Training
  - 20% Validation

---

## 🧠 Model: XGBoost Classifier

- **Base model**: `XGBClassifier` from `xgboost` library
- **Key Parameters**:
  - `use_label_encoder=False`
  - `eval_metric='mlogloss'`
  - `random_state=42`

### ⚙️ Compilation

- Objective: Multi-class classification
- Evaluation Metric: **Log Loss (mlogloss)**

---

## 🏋️ Training

- Fit model on training data
- Made predictions on validation data
- Evaluated with:
  - `classification_report` (precision, recall, F1)
  - `confusion_matrix`

---

## 📈 Evaluation

- Output: Full classification report for each forest type
- Visualized Confusion Matrix using `matplotlib` and `ConfusionMatrixDisplay`

---

## 📊 Results

- Achieved solid performance on validation data
- Confusion matrix revealed class distribution and model accuracy
- Future improvements may include:
  - Feature engineering
  - Full dataset training
  - Hyperparameter tuning (GridSearch or Optuna)

---

## 🧰 Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - XGBoost
  - Pandas
  - NumPy
  - scikit-learn
  - Matplotlib
  - Seaborn

---

## 👨‍💻 Author

**Omar Al Zoghbi**

---

## 📎 Optional Enhancements

- 📊 Add feature importance plots
- 🧪 Include hyperparameter tuning
- 💾 Provide `requirements.txt` for full reproducibility



## 📦 Dataset

The dataset includes:

- `train.csv` — Training data with features and `Cover_Type` labels.
- `test.csv` — Test data with the same features but without labels.
- `sample_submission.csv` — Submission format example.

📌 [Kaggle competition link](https://www.kaggle.com/competitions/tabular-playground-series-dec-2021)

---

## ⚙️ Environment Setup

```python
# Upload kaggle.json API key
from google.colab import files
files.upload()

# Setup Kaggle credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and unzip the competition data
!kaggle competitions download -c tabular-playground-series-dec-2021
!unzip tabular-playground-series-dec-2021.zip

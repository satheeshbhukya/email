# Email Click Prediction Project

This project predicts whether a user will click a link in an email campaign based on user and email features. The goal is to help optimize email marketing strategies and improve click-through rates (CTR).

---

## Dataset

The dataset contains three CSV files:
- `email_table.csv`: Email metadata including email type, version, send time, etc.
- `email_opened_table.csv`: Info on whether a user opened the email.
- `link_clicked_table.csv`: Target variable indicating if the user clicked a link.

---

## Objective

Predict the likelihood (`link_clicked`) that a user clicks on a link from an email using classification models. The key metric used for evaluation is **ROC AUC Score**.

---

## Technologies Used

- Python, Pandas
- Scikit-learn
- XGBoost
- LightGBM,
- StackingClassifier
- SHAP (for explainability)
- Matplotlib/Seaborn (for visualizations)

---

## Pipeline Overview

### 1. Data Preprocessing & Merging. 
- Calculated whether each email_id was opened or clicked from email_opened_table and link_clicked_table with email_table ,and added email_opened and 
  link_clicked binary columns to the main email table.
- Performed stratified train-test split (80-20).

### 2. Feature Engineering
- **Time-based features**: Time of day, weekday, hour–weekday interaction
- **Purchase binning**: Binned user purchases into `low`, `medium`, `high`
- **Content interaction**: Created text-version interaction, email length indicators
- **Polynomial features**: Squared past purchases
- **Encodings**:
  - Frequency Encoding: `user_country`, `weekday`
  - Target Encoding: `user_country`, `weekday`
  - Label Encoding: Categorical columns

### 3. Modeling
- **Base model**: XGBoostClassifier with class imbalance handling using `scale_pos_weight`
- **Tuning**: Used `RandomizedSearchCV` to tune hyperparameters
- **Ensembling**: Attempted stacking with logistic regression as meta-learner

### 4. Evaluation
- **Base ROC AUC Score**: `0.7426`
- **Tuned Model ROC AUC**: `0.7393`
- **Stacking Ensemble ROC AUC**: `0.7392`
- **Cross-Validation Mean ROC AUC**: `0.7363`

---

## Insights

### 1. CTR by Email Type & Personalization
| Email Text | Version       | CTR      |
|------------|----------------|----------|
| Long       | Generic        | 1.37%    |
| Long       | Personalized   | 2.34%    |
| Short      | Generic        | 1.65%    |
| Short      | Personalized   | **3.12%** |

✅ **Personalized + Short emails** performed best.

### 2. CTR Improvement
CTR improved by **2.07%** using feature engineering and modeling optimizations.

---

## Future Work

- Improve class imbalance handling using techniques like SMOTE or ADASYN
- Use LightGBM or deep learning models for additional performance gains
- Incorporate user behavior data (e.g., previous email engagement trends)

---

## Author

Satheesh Bhukya  
[GitHub](https://github.com/satheeshbhukya) | [Email](mailto:satheeshbhukya@gmail.com) |

---

## License

MIT License

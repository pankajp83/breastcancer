
# Breast Cancer Prediction using Machine Learning

## Project Overview
This project aims to build a machine learning model to predict breast cancer diagnoses as either malignant (M) or benign (B) based on patient diagnostic data. By leveraging data analysis and classification techniques, the model supports early detection and better decision-making in healthcare.

---

## Dataset
- **Source:** Breast Cancer dataset (uploaded)
- **Features:** 30 numerical features related to tumor characteristics (e.g., radius, texture, smoothness).
- **Target Variable:** Diagnosis (`M` for malignant, `B` for benign).

---

## Workflow
1. **Data Preprocessing:**
   - Removed unnecessary columns (`id`).
   - Encoded the target variable (`M` -> 1, `B` -> 0).
   - Standardized the features for optimal performance.
2. **Exploratory Data Analysis:**
   - Generated a correlation heatmap for feature analysis.
   - Visualized data distributions and correlations.
3. **Model Development:**
   - Trained a Random Forest Classifier.
   - Evaluated using precision, recall, F1-score, and ROC-AUC metrics.
4. **Model Evaluation:**
   - Achieved 96% accuracy with excellent ROC-AUC of 0.98.
   - Balanced performance for both malignant and benign predictions.

---

## Results

### **Classification Report**
```
              precision    recall  f1-score   support

           0       0.96      0.99      0.97        71
           1       0.98      0.93      0.95        43

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114
```

### **Confusion Matrix**
![Confusion Matrix](/images/confusion_matrix.png)

### **ROC-AUC Curve**
![ROC-AUC Curve](/images/roc_auc_curve.png)

---

## Conclusion
The machine learning model is highly accurate and effective at predicting breast cancer diagnoses. This tool can aid in clinical decision-making and early detection.

---

## Files
- **Notebook:** Contains all analysis and code.
- **Dataset:** Provided as input.
- **Images:** Generated from the analysis.

---

## Future Improvements
- Implement additional models like Gradient Boosting (XGBoost, LightGBM).
- Use deep learning (CNN) for image-based datasets such as mammograms.
- Develop a user-friendly interface for predictions (e.g., Streamlit or Flask).

---

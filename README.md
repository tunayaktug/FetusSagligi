# FetusSagligi
A machine learning project aimed at predicting fetal health, supported by Random Forest and LSTM algorithms, involving data merging and analytical processes.


# 🧠 Fetal Health Prediction Using Machine Learning & Deep Learning

## 📌 Overview
This project focuses on predicting fetal health status using a combination of **machine learning** and **deep learning** models. The primary objective is to classify fetal conditions as Normal, Suspect, or Pathological using a comprehensive dataset formed by merging two different sources. The system leverages both **Random Forest** and **LSTM** models to capture non-linear patterns and temporal trends in the data.

## 🧪 Technologies Used
- 🧹 **Preprocessing**: Null value handling, normalization, label encoding
- 📊 **Data**: `fetal_health.csv`, `train_dataset.csv`, combined into `merged_dataset.csv`
- 🤖 **Models**:
  - **Random Forest Classifier**: For baseline traditional classification
  - **LSTM Neural Network**: For temporal pattern recognition
- 📉 **Evaluation Metrics**: Accuracy, Confusion Matrix, ROC-AUC, F1-Score

## 📂 File Structure
├── fetal_health.csv # Original dataset 1
├── train_dataset.csv # Original dataset 2
├── merged_dataset.csv # Final merged dataset
├── FetüsSağlığıVeriAnalizi.ipynb # Full analysis & model notebook
├── README.txt # Contributor info


## 📈 Key Features
- Dataset merging and cleaning pipeline
- Comparative model analysis (Random Forest vs LSTM)
- Visual analytics (e.g., correlation heatmaps, ROC curves)
- Model tuning and performance logging


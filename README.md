# 📰 Fake Real News Detection
A Machine Learning Model for Detecting Fake and Real News.

## 📌 Project Description
This project aims to classify news articles as **fake** or **real** using Machine Learning. The dataset was preprocessed and trained using different models like:
- ✅ **Logistic Regression**
- ✅ **Random Forest Classifier** (Final Model)

## 🚀 Features
- Uses **TF-IDF Vectorization** for text processing  
- Compares multiple ML models to find the best one  
- Outputs whether a news article is **real** or **fake**  

## 🔧 How to Run the Model
1. Download `Fake_Real_News_model.ipynb`
2. Open in **Google Colab** or **Jupyter Notebook**
3. Install required libraries:
   ```python
   pip install pandas numpy sklearn

📉 Model Performance
Final Model: RandomForestClassifier
Accuracy: 71% (Needs Improvement)


📊 Classification Report (Test Set)
              precision    recall  f1-score   support

           0       0.74      0.59      0.65      4252
           1       0.69      0.81      0.74      4728

    accuracy                           0.71      8980
   macro avg       0.71      0.70      0.70      8980
weighted avg       0.71      0.71      0.70      8980


⚠️ Limitations
Performance can be improved with hyperparameter tuning
Using Deep Learning (LSTMs, BERT) could enhance results
Dataset size and quality may impact accuracy


🔥 Future Improvements
Train with Deep Learning models (LSTMs, Transformers)
Apply Hyperparameter tuning
Improve data preprocessing


📂 Files in This Repository
File Name	Description
Fake_Real_News_model.ipynb	Jupyter Notebook with model training and evaluation
Fake_Real_News_model.pkl	Saved trained model for reuse


📖 How to Use the .pkl Model?
To load and use the trained model:

import pickle
# Load the model
with open("Fake_Real_News_model.pkl", "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully:", type(model))


👨‍💻 Author
Syed Omer Hussaini




# 🏥 Multi-Disease Prediction System

Machine learning-powered web app for predicting **Diabetes**, **Heart Disease**, and **Parkinson's Disease** using trained ML models and Streamlit.

---

## ✨ Features

| Disease | Input Features | Model | Predictions |
|---------|---|---|---|
| **Diabetes** | 8 health metrics | SVM (GridSearchCV) | Disease risk with confidence |
| **Heart Disease** | 12 parameters | SVM (Pipeline) | Disease risk with confidence |
| **Parkinson's** | 15 voice features | Random Forest | Disease risk with confidence |

---

## 📊 Model Performance

| Model | Accuracy | Recall | Precision | F1-Score |
|-------|----------|--------|-----------|----------|
| Diabetes | 72% | 74% | Balanced | 0.73 |
| Heart Disease | High | Optimized | High | Strong |
| Parkinson's | **95%** | **97%** | High | **0.97** |

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn, XGBoost
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Deployment**: joblib (model persistence)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run streamlit_app.py
```

App opens at `http://localhost:8501`

---

## 📁 Project Structure

```
Multiple Disease/
├── streamlit_app.py          # Main application
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── data/                     # Datasets
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinson.csv
├── models/                   # Pre-trained models
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   └── parkinsons_model.pkl
└── notebooks/                # Training notebooks
    ├── Diabetes.ipynb
    ├── Heart Disease.ipynb
    └── Parkinson.ipynb
```

---

## 💡 How to Use

1. Select disease from sidebar
2. Enter patient health metrics
3. Click "Predict" button
4. View prediction + confidence score

---

## ⚠️ Important Disclaimer

**Educational purposes only.** Not a substitute for professional medical diagnosis. Always consult healthcare professionals.

---

## 👤 Author
**Anushka Sharma**
Created as an educational ML project for disease prediction analysis.

---

## � License

This project is licensed under the **MIT License** - see below for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## �📝 Notes

- Models are conservative (avoid false positives)
- Lower confidence = recommendation for further testing
- All models optimized for high recall (minimize false negatives)

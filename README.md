# 🏡 House Prices Prediction — Linear Regression

This project predicts the sale prices of houses based on various features using **Linear Regression**. It's based on the Kaggle competition ["House Prices: Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

---

## 📁 Dataset

- **Source:** Kaggle
- **Files Used:**
  - `train.csv` — training data
  - `test.csv` — test data for predictions
- **Target Variable:** `SalePrice`
- **Features:** 79 (numerical + categorical)

---

## 🚀 Technologies Used

- **Python 3.x**
- **Pandas, NumPy** — Data manipulation
- **Matplotlib, Seaborn** — Visualization
- **Scikit-learn** — Modeling and evaluation

---

## Project Structure
house-price-predictor/
- ├── data/
- │   ├── train.csv
- │   └── test.csv
- ├── house_price_predictor.py
- ├── submission.csv
- ├── requirements.txt
- └── README.md


## 🧠 Project Workflow

1. **Data Loading & Exploration**
2. **Data Cleaning**
   - Dropped high-missing-value columns
   - Handled null values
3. **Feature Engineering**
   - One-hot encoded categorical variables
4. **Model Training**
   - Trained a `LinearRegression` model
5. **Model Evaluation**
   - RMSE & R² Score on validation set
6. **Prediction on Test Set**
   - Created `submission.csv` for Kaggle

---

## 📊 Model Performance

- **RMSE:** ~30,000 (on test split)
- **R² Score:** ~0.84

---

## 📦 How to Run

1. Clone this repo or download the files.
2. Place `train.csv` and `test.csv` inside a `data/` folder.
3. Run the script:
   ```bash
   python house_price_predictor.py

## 👤 Author

**Vikas Shelar**  
📍 Mumbai, India  
📧 [vickyshelar96k@gmail.com](mailto:vickyshelar96k@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/vikas-shelar-1508s)  

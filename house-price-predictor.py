import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Load train data
train_df = pd.read_csv('data/train.csv')

# Drop columns with too many missing values
train_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)

# Fill missing values
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
cat_cols = train_df.select_dtypes(include='object').columns
for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)

# One-hot encode
train_df = pd.get_dummies(train_df)

# Drop Id
train_df.drop('Id', axis=1, inplace=True)

# Train/Test Split
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:\nRMSE: {rmse:.2f}\nR² Score: {r2:.4f}")

# Load test data
test_df = pd.read_csv('data/test.csv')
test_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)
cat_cols_test = test_df.select_dtypes(include='object').columns
for col in cat_cols_test:
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# One-hot encode and align with training features
test_df = pd.get_dummies(test_df)
test_df = test_df.reindex(columns=X.columns, fill_value=0)

# Predict test data
test_preds = model.predict(test_df)

# Save to submission.csv
submission = pd.DataFrame({
    'Id': pd.read_csv('data/test.csv')['Id'],
    'SalePrice': test_preds
})
submission.to_csv('submission.csv', index=False)
print("submission.csv created successfully.")

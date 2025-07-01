import pandas as pd
import joblib
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

start = time.time()

# Load dataset
df = pd.read_csv("data/customer_churn_dataset-training-master.csv")
print("Dataset loaded:", df.shape)

# Drop rows with missing values (optional, quick cleanup)
df.dropna(inplace=True)

# Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a lightweight XGBoost model
model = XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss', verbosity=0)
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/churn_model.pkl")

end = time.time()
print(f"Model trained and saved in {end - start:.2f} seconds")

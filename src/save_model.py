
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "california_housing.csv"
model_folder = project_root / "models"
model_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

model_path = model_folder / "house_price_model.pkl"
joblib.dump(model, model_path)

print("Model saved successfully!")
print("Saved to:", model_path)

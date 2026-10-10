
from pathlib import Path
from sklearn.datasets import fetch_california_housing

project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "data" / "raw"
data_folder.mkdir(parents=True, exist_ok=True)

housing = fetch_california_housing(as_frame=True)
df = housing.frame

df.to_csv(data_folder / "california_housing.csv", index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

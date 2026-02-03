import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_excel("data.xlsx")

# Date conversion
df["OPEN_DATE"] = pd.to_datetime(df["OPEN_DATE"], errors="coerce")
df["HIRED_DATE"] = pd.to_datetime(df["HIRED_DATE"], errors="coerce")

# Target
df["time_to_hire_days"] = (df["HIRED_DATE"] - df["OPEN_DATE"]).dt.days
df = df[df["time_to_hire_days"] > 0]

# Leakage removal
df = df.drop(columns=["OPEN_DATE", "HIRED_DATE"])

# Prepare features
X = df.drop(columns=["time_to_hire_days", "OPEN_DATE", "HIRED_DATE"])
y = df["time_to_hire_days"]

# Remove datetime columns explicitly
X = X.select_dtypes(exclude=["datetime64[ns]"])

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))

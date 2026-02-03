
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def prepare_data():
    df = pd.read_excel("data.xlsx")

    # Convert dates
    df["OPEN_DATE"] = pd.to_datetime(df["OPEN_DATE"], errors="coerce")
    df["HIRED_DATE"] = pd.to_datetime(df["HIRED_DATE"], errors="coerce")

    # Create target
    df["time_to_hire_days"] = (df["HIRED_DATE"] - df["OPEN_DATE"]).dt.days
    df = df[df["time_to_hire_days"] > 0]

    X = df.drop(columns=["time_to_hire_days", "OPEN_DATE", "HIRED_DATE"])
    y = df["time_to_hire_days"]

    return X, y


def test_target_exists_and_positive():
    _, y = prepare_data()
    assert y.notnull().all()
    assert (y > 0).all()


def test_model_mae_reasonable():
    X, y = prepare_data()

    X = pd.get_dummies(X, drop_first=True)
    X = X.select_dtypes(include=["number"])


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    assert mae < 30


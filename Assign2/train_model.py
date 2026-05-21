import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import pickle

def train_and_save_models(csv_path="airbnb_data.csv", test_size=0.25, n_estimators=200):
    df = pd.read_csv(csv_path)

    data = df.drop(columns=["price_eur", "occupancy"])
    real_price = df["price_eur"]
    real_occ = df["occupancy"]

    quant_columns = ["month","day_of_week","lead_days","competition","rooms","review_score","is_weekend","is_holiday","has_parking","has_pool"]
    qual_columns = ["event","location","district_type","property_type"]

    data = data[quant_columns + qual_columns] 

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preproc = ColumnTransformer(transformers=[
        ("num", numeric_transformer, quant_columns),
        ("cat", categorical_transformer, qual_columns)])

    X_train, X_test, y_price_train, y_price_test, y_occ_train, y_occ_test = train_test_split(
        data, real_price, real_occ, test_size=test_size, random_state=7
    )

    price_mdl = Pipeline(steps=[("pre", preproc), ("reg", RandomForestRegressor(n_estimators=n_estimators, random_state=7))])
    price_mdl.fit(X_train, y_price_train)

    occ_mdl = Pipeline(steps=[("pre", preproc), ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=7))])
    occ_mdl.fit(X_train, y_occ_train)

    output = {
        "price_mdl":    price_mdl,
        "occ_mdl":      occ_mdl,
        "X_test":       X_test,
        "y_price_test": y_price_test,
        "y_occ_test":   y_occ_test,
    }

    with open("airbnb_models.pkl", "wb") as f:
        pickle.dump(output, f)
        
    return output

if __name__ == "__main__":
    print("Training models...")
    train_and_save_models()
    print("Models successfully saved to airbnb_models.pkl")
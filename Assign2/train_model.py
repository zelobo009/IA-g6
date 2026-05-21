
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import pickle



df = pd.read_csv("airbnb_data.csv")

data = df.drop(columns=["price_eur", "occupancy"])
real_price = df["price_eur"]
real_occ = df["occupancy"]

quant_columns = ["month","day_of_week","lead_days","competition","rooms","review_score","is_weekend","is_holiday","has_parking","has_pool"]
qual_columns = ["event","location","district_type","property_type"]

data = data[quant_columns + qual_columns] 

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore");

preproc = ColumnTransformer(transformers=[
    ("num",numeric_transformer, quant_columns),
    ("cat",categorical_transformer, qual_columns)])

X_train, X_test, y_price_train, y_price_test, y_occ_train, y_occ_test = train_test_split(data, real_price, real_occ, test_size=0.25, random_state=7)

price_mdl = Pipeline(steps=[("preprocessor", preproc), ("regression", RandomForestRegressor(n_estimators=200, random_state=7))])

price_mdl.fit(X_train, y_price_train)

occ_mdl = Pipeline(steps=[("preprocessor", preproc), ("classifier", RandomForestClassifier(n_estimators=200, random_state=7))])

occ_mdl.fit(X_train, y_occ_train)


y_price_pred_t = price_mdl.predict(X_test)
print("── Price model ──────────────────────────")
print(f"  MAE : €{mean_absolute_error(y_price_test, y_price_pred_t):.2f}")
print(f"  R²  : {r2_score(y_price_test, y_price_pred_t):.3f}")

y_occ_pred_t = occ_mdl.predict(X_test)
print("\n── Occupancy model ──────────────────────")
print(f"  Accuracy : {accuracy_score(y_occ_test, y_occ_pred_t):.3f}")
print(classification_report(y_occ_test, y_occ_pred_t, target_names=["Not booked","Booked"]))



with open("airbnb_models.pkl", "wb") as f:
    pickle.dump({
        "price_mdl": price_mdl,
        "occ_mdl":   occ_mdl,
        "X_test":    X_test,
        "y_price_test": y_price_test,
        "y_occ_test":   y_occ_test,
    }, f)
    
print("Models saved to airbnb_models.pkl")

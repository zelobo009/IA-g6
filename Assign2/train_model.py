
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



df = pd.read_csv("airbnb_data.csv")

data = df.drop(collums=["price_eur", "occupancy"])
real_price = df["price_eur"]
real_occ = df["occupancy"]

quant_collums = ["month","day_of_week","lead_days","competition","rooms","review_score","is_weekend","is_holiday","has_parking","has_pool"]
qual_collums = ["event","location","district_type","property_type"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore");

preproc = ColumnTransformer(transformers=[
    ("num",numeric_transformer, quant_collums),
    ("cat",categorical_transformer, qual_collums)])

X_train, X_test, y_price_train, y_price_test, y_occ_train, y_occ_test = train_test_split(data, real_price, real_occ, test_size=0.25, random_state=7)

price_mdl = Pipeline(steps=[("preprocessor", preproc), ("regression", RandomForestRegressor(n_estimators=100, random_state=7))])

price_mdl.fit(X_train, y_price_train)

occ_mdl = Pipeline(steps=[("preprocessor", preproc), ("class", RandomForestClassifier(n_estimators=100, random_state=7))])

occ_mdl.fit(X_train, y_occ_train)

price_mdl


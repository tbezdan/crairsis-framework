folder = "H:/Projekti/Odobreni/2023 Prizma/data/IPB/"
file = "COVID.csv"
feature_mapping = "feature_mapping.csv"

import pandas as pd

data = pd.read_csv(folder + file)
data["Datetime"] = pd.to_datetime(data["Datetime"])
dates = data["Datetime"]
data.set_index("Datetime", inplace=True)

lag_features = pd.read_csv(folder + feature_mapping)
lag_features = lag_features[
    lag_features["Category"].isin(
        ["Mobility", "Governmental measures", "Meteo", "Active cases"]
    )
]["Feature"].tolist()

num_days = (
    3  # mora manje dana zato sto imamo malo podataka za prekovid (0) i postkovid (2)
)
for f in lag_features:
    for i in range(1, num_days + 1):
        data[f"{f}_lag_{i}"] = data[f].shift(freq=f"{i}D")

data = data.dropna()

data.to_csv(folder + "lagged - " + file, index=True)

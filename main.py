import pandas as pd
import numpy as np

df = pd.read_csv("clima.csv")

df["FECHA"] = pd.to_datetime(df["FECHA"])

print(df.head())

df["TMIN"] = pd.to_numeric(df["TMIN"], errors='coerce')
df["EVAP"] = pd.to_numeric(df["EVAP"], errors='coerce')
df["PRECIP"] = pd.to_numeric(df["PRECIP"], errors='coerce')








df["HELADA"] = (df["TMIN"] <= 0).astype(int)










df["SEQUIA"] = (
    df["EVAP"] > 1.5 * df["PRECIP"]
).astype(float)











umbral_lluvia = df["PRECIP"].quantile(0.90)

df["INUNDACION"] = (
    df["PRECIP"] > umbral_lluvia
).astype(int)












df["RIESGO"] = (
    (df["HELADA"] == 1) |
    (df["SEQUIA"] == 1) |
    (df["INUNDACION"] == 1)
).astype(int)


print(df["RIESGO"])
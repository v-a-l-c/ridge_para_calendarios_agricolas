import pandas as pd
import numpy as np

df = pd.read_csv("clima.csv")

df["FECHA"] = pd.to_datetime(df["FECHA"])

print(df.head())

df["TMIN"] = pd.to_numeric(df["TMIN"], errors='coerce')
df["EVAP"] = pd.to_numeric(df["EVAP"], errors='coerce')
df["PRECIP"] = pd.to_numeric(df["PRECIP"], errors='coerce')






df["HELADA"] = (
    df["TMIN"] <= 0
).astype(int)


df["MES"] = df["FECHA"].dt.month
umbral_sequia = df.groupby("MES")["PRECIP"].transform(lambda x: x.quantile(0.3))
df["SEQUIA"] = (df["PRECIP"] < umbral_sequia).astype(int)

"""
df["SEQUIA"] = (
    df["EVAP"] > 1.5 * df["PRECIP"]
).astype(int)

umbral_inundacion = (
    df["PRECIP"]
    .quantile(0.90)
)
"""



umbral_inundacion = (
    df["PRECIP"]
    .quantile(0.90)
)



df["INUNDACION"] = (
    df["PRECIP"] > umbral_inundacion
).astype(int)

df["RIESGO"] = (
    (df["HELADA"] == 1)
    |
    (df["SEQUIA"] == 1)
    |
    (df["INUNDACION"] == 1)
).astype(int)













df["DIA_AÑO"] = df["FECHA"].dt.dayofyear




df["dia_sin"] = np.sin(2 * np.pi * df["DIA_AÑO"] / 365)
df["dia_cos"] = np.cos(2 * np.pi * df["DIA_AÑO"] / 365)


df["PRECIP_lag1"] = (
    df["PRECIP"].shift(1)
)

df["TMAX_lag1"] = (
    df["TMAX"].shift(1)
)

df["TMIN_lag1"] = (
    df["TMIN"].shift(1)
)

df["EVAP_lag1"] = (
    df["EVAP"].shift(1)
)

df = df.dropna()




features = [
    "PRECIP",
    "EVAP",
    "TMAX",
    "TMIN",
    "dia_sin",
    "dia_cos",
    "PRECIP_lag1",
    "TMAX_lag1",
    "TMIN_lag1",
    "EVAP_lag1"
]

X = df[features]
y = df["RIESGO"]










split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]



####--------------------NORMALIZACIÓN----------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


####---------------KERNEL LINEAL--------------------------------


from sklearn.kernel_ridge import KernelRidge

modelo_lineal = KernelRidge(
    alpha=1.0,
    kernel="linear"
)

modelo_lineal.fit(
    X_train,
    y_train
)

pred_lineal = (
    modelo_lineal
    .predict(X_test)
)

pred_lineal = (
    pred_lineal > 0.5
).astype(int)


####---------------------KERNEL NO LINEAL----------------


modelo_rbf = KernelRidge(
    alpha=1.0,
    kernel="rbf",
    gamma=0.1
)

modelo_rbf.fit(
    X_train,
    y_train
)

pred_rbf = (
    modelo_rbf
    .predict(X_test)
)

pred_rbf = (
    pred_rbf > 0.5
).astype(int)

####----------------------EVALUACIÓN-----------------------



from sklearn.metrics import accuracy_score

print("Kernel lineal:")
print(
    accuracy_score(
        y_test,
        pred_lineal
    )
)

print("Kernel RBF:")
print(
    accuracy_score(
        y_test,
        pred_rbf
    )
)




print(df["HELADA"].mean())
print(df["SEQUIA"].mean())
print(df["INUNDACION"].mean())


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(y_test, pred_lineal))
print(confusion_matrix(y_test, pred_rbf))

print(classification_report(y_test, pred_lineal))
print(classification_report(y_test, pred_rbf))




####---------------ELEGIR FECHA

X_scaled = scaler.transform(
    df[features]
)

df["RIESGO_SCORE"] = (
    modelo_rbf.predict(X_scaled)
)

riesgo = modelo_rbf.predict(X_scaled)



#FECHA segura cercana


from datetime import timedelta

def fecha_optima(
    fecha_objetivo,
    ventana_dias=20
):

    fecha_objetivo = pd.to_datetime(
        fecha_objetivo
    )

    inicio = (
        fecha_objetivo
        - timedelta(days=ventana_dias)
    )

    fin = (
        fecha_objetivo
        + timedelta(days=ventana_dias)
    )

    candidatos = df[
        (df["FECHA"] >= inicio)
        &
        (df["FECHA"] <= fin)
    ].copy()

    # distancia temporal
    candidatos["DISTANCIA"] = (
        candidatos["FECHA"]
        - fecha_objetivo
    ).abs().dt.days

    # ordenar:
    candidatos = candidatos.sort_values(
        by=[
            "RIESGO_SCORE",
            "DISTANCIA"
        ]
    )

    return candidatos.iloc[0]


#UNA FECHA
mejor = fecha_optima(
    "2024-06-24",
    ventana_dias=20
)

print(mejor[
    [
        "FECHA",
        "RIESGO_SCORE",
        "PRECIP",
        "TMAX",
        "TMIN"
    ]
])

#INTERVALO
def mejor_fecha_intervalo(
    fecha_inicio,
    fecha_fin
):

    inicio = pd.to_datetime(
        fecha_inicio
    )

    fin = pd.to_datetime(
        fecha_fin
    )

    candidatos = df[
        (df["FECHA"] >= inicio)
        &
        (df["FECHA"] <= fin)
    ]

    mejor = candidatos.loc[
        candidatos[
            "RIESGO_SCORE"
        ].idxmin()
    ]

    return mejor

resultado = mejor_fecha_intervalo(
    "2025-04-03",
    "2025-04-15"
)

print(resultado)
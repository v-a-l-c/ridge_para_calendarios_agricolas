import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Cargamos el histórico climático
df = pd.read_csv("clima.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])


# Aseguramos que las variables críticas sean numéricas
cols_numericas = ["TMIN", "TMAX", "EVAP", "PRECIP"]
for col in cols_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# Definimos de umbrales y etiquetas de riesgo (Clasificación binaria implícita)
df["HELADA"] = (df["TMIN"] <= 0).astype(int)

# Sequía basada en el percentil 30 mensual de precipitación
df["MES"] = df["FECHA"].dt.month
umbral_sequia = df.groupby("MES")["PRECIP"].transform(lambda x: x.quantile(0.3))
df["SEQUIA"] = (df["PRECIP"] < umbral_sequia).astype(int)

# Inundación basada en el percentil 90 global de precipitación
umbral_inundacion = df["PRECIP"].quantile(0.90)
df["INUNDACION"] = (df["PRECIP"] > umbral_inundacion).astype(int)


# Target lógico: 1 si ocurre al menos un evento adverso, 0 si es seguro
df["RIESGO"] = ((df["HELADA"] == 1) | (df["SEQUIA"] == 1) | (df["INUNDACION"] == 1)).astype(int)

# Variables de ciclicidad temporal (Transformación del día del año)
df["DIA_AÑO"] = df["FECHA"].dt.dayofyear
df["dia_sin"] = np.sin(2 * np.pi * df["DIA_AÑO"] / 365)
df["dia_cos"] = np.cos(2 * np.pi * df["DIA_AÑO"] / 365)

# Creación de variables rezagadas (Lag features de 1 día de desfase)
for col in cols_numericas:
    df[f"{col}_lag1"] = df[col].shift(1)

# Eliminar filas con nulos resultantes de los rezagos o conversiones
df = df.dropna().reset_index(drop=True)


####---------------DIVISIÓN Y ESCALADO DE DATOS--------------------------------

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



# Split secuencial (80% entrenamiento, 20% test) para no romper el orden temporal
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Normalización de variables numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


####---------------ENTRENAMIENTO DE MODELOS--------------------------------

# --- Kernel Ridge Regression ---
krr_lineal = KernelRidge(alpha=1.0, kernel="linear")
krr_lineal.fit(X_train_scaled, y_train)
pred_krr_lineal = krr_lineal.predict(X_test_scaled)

krr_rbf = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)
krr_rbf.fit(X_train_scaled, y_train)
pred_krr_rbf = krr_rbf.predict(X_test_scaled)



# --- Support Vector Regression ---
svr_lineal = SVR(kernel="linear", C=1.0, epsilon=0.1)
svr_lineal.fit(X_train_scaled, y_train)
pred_svr_lineal = svr_lineal.predict(X_test_scaled)

svr_rbf = SVR(kernel="rbf", C=1.0, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train_scaled, y_train)
pred_svr_rbf = svr_rbf.predict(X_test_scaled)


####----------------------EVALUACIÓN-----------------------



def evaluar_modelo(nombre, y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    print(f"{nombre:15} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

print("\n Evaluación de modelos:")
evaluar_modelo("KRR Lineal", y_test, pred_krr_lineal)
evaluar_modelo("KRR RBF", y_test, pred_krr_rbf)
evaluar_modelo("SVR Lineal", y_test, pred_svr_lineal)
evaluar_modelo("SVR RBF", y_test, pred_svr_rbf)


###-----------SELECCIÓN DE FECHAS------------------------

# Usamos el modelo KRR RBF para calcular el Score de Riesgo global
X_all_scaled = scaler.transform(df[features])
df["RIESGO_SCORE"] = krr_rbf.predict(X_all_scaled)

def obtener_fecha_optima(fecha_objetivo, ventana_dias=20):
    fecha_obj = pd.to_datetime(fecha_objetivo)
    inicio = fecha_obj - timedelta(days=ventana_dias)
    fin = fecha_obj + timedelta(days=ventana_dias)
    
    candidatos = df[(df["FECHA"] >= inicio) & (df["FECHA"] <= fin)].copy()
    
    if candidatos.empty:
        return "No hay datos para el rango seleccionado."
        
    # Calcular penalización por distancia en días a la fecha deseada
    candidatos["DISTANCIA"] = (candidatos["FECHA"] - fecha_obj).abs().dt.days
    
    # Priorizar menor score de riesgo; a igual score, la fecha más cercana
    candidatos = candidatos.sort_values(by=["RIESGO_SCORE", "DISTANCIA"])
    return candidatos.iloc[0]

def mejor_fecha_intervalo(fecha_inicio, fecha_fin):
    """Encuentra la fecha con menor riesgo absoluto dentro de un rango dado."""
    inicio, fin = pd.to_datetime(fecha_inicio), pd.to_datetime(fecha_fin)
    candidatos = df[(df["FECHA"] >= inicio) & (df["FECHA"] <= fin)]
    
    if candidatos.empty:
        return "No hay datos para el intervalo seleccionado."
        
    mejor_idx = candidatos["RIESGO_SCORE"].idxmin()
    return candidatos.loc[mejor_idx]

print("\nOPTIMIZACIÓN DE FECHA INDIVIDUAL:")
mejor_individual = obtener_fecha_optima("2024-06-24", ventana_dias=20)
print(mejor_individual[["FECHA", "RIESGO_SCORE", "PRECIP", "TMAX", "TMIN"]])

print("\nOPTIMIZACIÓN POR INTERVALO:")
mejor_intervalo = mejor_fecha_intervalo("2025-04-03", "2025-04-15")
print(mejor_intervalo[["FECHA", "RIESGO_SCORE", "PRECIP", "TMAX", "TMIN"]])

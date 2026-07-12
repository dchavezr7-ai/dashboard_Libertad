# ============================================================
# CELDA A AGREGAR EN TU COLAB (después de tener `pdf` cargado)
# Exporta el dataset histórico que alimentará el Dashboard Streamlit
# ============================================================
COLUMNAS_DASHBOARD = [
    "ANIO", "MES", "PROVINCIA", "DISTRITO", "TIPO_DELITO",
    "POBLACION", "DENSIDAD_POB", "LAG_1", "LAG_3",
    "PROMEDIO_MOVIL_3M", "VAR_INTERANUAL",
    "CANTIDAD_DELITOS", "TASA_DELITOS_10K", "NIVEL_RIESGO",
]

historico_dashboard = pdf[COLUMNAS_DASHBOARD].copy()
historico_dashboard.to_csv("historico_dashboard.csv", index=False, encoding="utf-8-sig")

print(f"✅ Archivo generado: historico_dashboard.csv ({len(historico_dashboard):,} filas)")

try:
    from google.colab import files
    files.download("historico_dashboard.csv")
except ImportError:
    print("ℹ️ Fuera de Colab: el archivo ya quedó guardado localmente.")

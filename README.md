# Dashboard de Riesgo Delictivo — La Libertad

## 1. Qué archivos necesitas juntar en la carpeta del proyecto

```
tu_proyecto/
├── app.py
├── requirements.txt
├── packages.txt
├── modelos_entrenados.zip        <- lo genera tu notebook (celda 41-42), NO lo edites
└── historico_dashboard.csv       <- nuevo, generado con la celda de abajo
```

`app.py` descomprime automáticamente `modelos_entrenados.zip` la primera vez que corre
la app (no necesitas descomprimirlo tú a mano, aunque también funciona si ya está
descomprimido en una carpeta `modelos_entrenados/`).

## 2. Paso pendiente en tu Colab

Tu notebook ya exporta los modelos correctamente (celdas 41-42, con
`.write().overwrite().save(...)` — es la forma correcta para modelos de PySpark
MLlib, **no uses pickle/joblib con estos objetos**, no son compatibles).

Lo único que falta es exportar el dataset histórico que alimenta los selectores
del dashboard. Agrega la celda de `celda_exportar_csv_colab.py` (adjunta) a tu
notebook, ejecútala, y descarga `historico_dashboard.csv`.

## 3. Ejecutar localmente

```bash
pip install -r requirements.txt
# Necesitas Java 17 instalado localmente para que PySpark funcione:
# Ubuntu/Debian: sudo apt install openjdk-17-jdk-headless
# Mac: brew install openjdk@17
streamlit run app.py
```

## 4. Desplegar en Streamlit Community Cloud

- Sube los 5 archivos de la carpeta a un repo de GitHub.
- El archivo `packages.txt` le indica a Streamlit Cloud que instale Java
  (`openjdk-17-jdk-headless`) — sin esto, PySpark falla en el servidor.
- `modelos_entrenados.zip` puede pesar varios MB; si supera el límite de GitHub/Streamlit
  considera Git LFS o alojarlo en Google Drive y descargarlo en `cargar_spark_y_modelos()`.

## 5. Qué hace cada pestaña

- **Predecir**: busca un registro histórico real (Provincia → Distrito → Tipo de
  delito → Año → Mes) y muestra la predicción del modelo comparada con el valor
  real observado.
- **Proyectar**: construye las variables (`LAG_1`, `LAG_3`, `PROMEDIO_MOVIL_3M`,
  `VAR_INTERANUAL`) para el mes siguiente al último dato disponible (o hasta 3 meses
  encadenados, con indicador de confianza decreciente), replicando exactamente la
  lógica de las celdas 47-48 de tu notebook.

"""
Dashboard interactivo — Predicción de Riesgo Delictivo (Región La Libertad)
Modelos: RandomForestClassifier (NIVEL_RIESGO) + RandomForestRegressor (TASA_DELITOS_10K)
Motor: PySpark MLlib (los modelos se cargan con la persistencia nativa de Spark, no pickle/joblib)
"""

import os
import sys
import zipfile
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------------
# FIX WINDOWS: asegura que el worker de PySpark use el MISMO intérprete de Python
# que está corriendo Streamlit, y que Spark se enlace solo a localhost. Sin esto,
# en Windows es común el error "Python worker failed to connect back" por el
# Firewall o por que Spark intenta lanzar un python distinto/inexistente.
# ----------------------------------------------------------------------------------
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ----------------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Riesgo Delictivo — La Libertad",
    page_icon="🚓",
    layout="wide",
)

RUTA_MODELOS = "modelos_entrenados"
RUTA_ZIP = "modelos_entrenados.zip"
RUTA_CSV = "historico_dashboard.csv"

# Años que corresponden al conjunto de PRUEBA del modelo (ver Notebook 2, evaluación
# final: "su desempeño se evalúa sobre 2025-2026"). 2022-2024 se usó para entrenar,
# así que no tiene sentido "predecir" sobre esos años en la pestaña histórica.
ANIO_MIN_PREDICCION = 2025

MESES_NOMBRE = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

COLOR_RIESGO = {"BAJO": "#2ecc71", "MEDIO": "#f39c12", "ALTO": "#e74c3c"}

RECOMENDACION_RIESGO = {
    "BAJO": "Mantener el patrullaje preventivo habitual y continuar el monitoreo rutinario del distrito.",
    "MEDIO": "Reforzar la vigilancia, aumentar la frecuencia de patrullajes y monitorear de cerca la evolución del distrito.",
    "ALTO": "Incrementar el patrullaje preventivo, reforzar los operativos policiales y priorizar recursos en el distrito.",
}

NIVEL_CONFIANZA = {
    0: ("Alta", "#2ecc71", "Calculado 100% con datos reales registrados."),
    1: ("Media", "#f39c12", "Usa 1 valor propio proyectado en el paso anterior (LAG_1)."),
    2: ("Baja", "#e74c3c", "Usa 2 o más valores proyectados encadenados; la incertidumbre se acumula."),
}


# ----------------------------------------------------------------------------------
# CARGA DE RECURSOS (cacheados para no recargar en cada interacción)
# ----------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Iniciando motor Spark y cargando modelos entrenados…")
def cargar_spark_y_modelos():
    """Descomprime (si hace falta) y carga el pipeline + los dos modelos finales.
    Se llama SOLO cuando hace falta predecir/proyectar (no al abrir el dashboard),
    para que la pestaña de Resumen cargue rápido sin esperar a que arranque Spark."""
    if not os.path.isdir(RUTA_MODELOS):
        if os.path.exists(RUTA_ZIP):
            with zipfile.ZipFile(RUTA_ZIP, "r") as z:
                z.extractall(RUTA_MODELOS)
        else:
            st.error(
                f"No se encontró la carpeta '{RUTA_MODELOS}' ni el archivo '{RUTA_ZIP}'. "
                "Sube el ZIP de modelos exportado desde tu notebook de Colab junto a este app.py."
            )
            st.stop()

    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel
    from pyspark.ml.feature import StringIndexerModel
    from pyspark.ml.classification import RandomForestClassificationModel
    from pyspark.ml.regression import RandomForestRegressionModel

    spark = (
        SparkSession.builder
        .appName("DashboardDelitosLaLibertad")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    feature_pipeline_model = PipelineModel.load(f"{RUTA_MODELOS}/feature_pipeline_model")
    label_indexer_model = StringIndexerModel.load(f"{RUTA_MODELOS}/label_indexer_model")
    modelo_clf = RandomForestClassificationModel.load(f"{RUTA_MODELOS}/randomforest_clasificacion")
    modelo_reg = RandomForestRegressionModel.load(f"{RUTA_MODELOS}/randomforest_regresion")

    etiquetas_riesgo = label_indexer_model.labels  # índice -> "BAJO"/"MEDIO"/"ALTO"
    return spark, feature_pipeline_model, modelo_clf, modelo_reg, etiquetas_riesgo


@st.cache_data(show_spinner="Cargando dataset histórico…")
def cargar_historico() -> pd.DataFrame:
    if not os.path.exists(RUTA_CSV):
        st.error(
            f"No se encontró '{RUTA_CSV}'. Genera y sube este archivo con la celda de exportación "
            "que agregaste a tu Colab (ver celda_exportar_csv_colab.py)."
        )
        st.stop()
    return pd.read_csv(RUTA_CSV)


# ----------------------------------------------------------------------------------
# LÓGICA DE PREDICCIÓN (usa el pipeline + modelos de Spark ya entrenados)
# ----------------------------------------------------------------------------------
def ejecutar_prediccion(_spark, _pipeline, _modelo_clf, _modelo_reg, etiquetas_riesgo, fila_features: dict):
    """
    fila_features debe traer EXACTAMENTE:
    POBLACION, DENSIDAD_POB, LAG_1, LAG_3, PROMEDIO_MOVIL_3M, VAR_INTERANUAL,
    DISTRITO (texto), TIPO_DELITO (texto)
    El pipeline se encarga de indexar DISTRITO/TIPO_DELITO y ensamblar el vector "features".
    """
    columnas_modelo = [
        "POBLACION", "DENSIDAD_POB", "LAG_1", "LAG_3",
        "PROMEDIO_MOVIL_3M", "VAR_INTERANUAL", "DISTRITO", "TIPO_DELITO",
    ]
    fila_limpia = {k: fila_features[k] for k in columnas_modelo}

    spark_df = _spark.createDataFrame([fila_limpia])
    transformado = _pipeline.transform(spark_df)

    pred_clf_row = _modelo_clf.transform(transformado).select("prediction").collect()[0]
    pred_reg_row = _modelo_reg.transform(transformado).select("prediction").collect()[0]

    idx_riesgo = int(pred_clf_row["prediction"])
    nivel_riesgo = etiquetas_riesgo[idx_riesgo] if idx_riesgo < len(etiquetas_riesgo) else "DESCONOCIDO"
    tasa_predicha = float(pred_reg_row["prediction"])
    return nivel_riesgo, tasa_predicha


def buscar_registro_previo(hist: pd.DataFrame, distrito: str, tipo_delito: str, anio: int, mes: int,
                            meses_atras: int = 0, anios_atras: int = 0):
    """Replica _registro_previo del notebook: retrocede meses_atras meses y/o anios_atras
    años desde (anio, mes) y busca el registro real correspondiente (para comparar)."""
    anio_obj, mes_obj = anio - anios_atras, mes - meses_atras
    while mes_obj <= 0:
        mes_obj += 12
        anio_obj -= 1
    fila = hist[
        (hist.DISTRITO == distrito) & (hist.TIPO_DELITO == tipo_delito) &
        (hist.ANIO == anio_obj) & (hist.MES == mes_obj)
    ]
    return fila.iloc[0] if not fila.empty else None


def variacion_pct(nuevo, anterior):
    if anterior is None or anterior == 0:
        return None
    return (nuevo - anterior) / anterior * 100


def calcular_features_mes_siguiente(hist: pd.DataFrame, distrito: str, tipo_delito: str):
    """Replica _proyectar_mes_siguiente del notebook: construye el vector del mes
    inmediato siguiente al último registrado, usando solo datos reales."""
    sub = hist[(hist.DISTRITO == distrito) & (hist.TIPO_DELITO == tipo_delito)].sort_values(["ANIO", "MES"])
    if sub.empty:
        return None

    ultimo = sub.iloc[-1]
    anio_ref, mes_ref = int(ultimo["ANIO"]), int(ultimo["MES"])
    mes_obj, anio_obj = mes_ref + 1, anio_ref
    if mes_obj > 12:
        mes_obj, anio_obj = 1, anio_obj + 1

    def cantidad_en(a, m):
        f = sub[(sub.ANIO == a) & (sub.MES == m)]
        return float(f.iloc[0]["CANTIDAD_DELITOS"]) if not f.empty else None

    lag_1 = cantidad_en(anio_ref, mes_ref)

    m3, a3 = mes_obj - 3, anio_obj
    while m3 <= 0:
        m3 += 12
        a3 -= 1
    lag_3 = cantidad_en(a3, m3)

    valores = []
    for k in range(1, 4):
        mk, ak = mes_obj - k, anio_obj
        while mk <= 0:
            mk += 12
            ak -= 1
        v = cantidad_en(ak, mk)
        if v is not None:
            valores.append(v)
    promedio_movil = sum(valores) / len(valores) if valores else None

    if lag_1 is None or lag_3 is None or promedio_movil is None:
        return None

    return {
        "ANIO": anio_obj, "MES": mes_obj,
        "PROVINCIA": ultimo["PROVINCIA"],
        "POBLACION": float(ultimo["POBLACION"]),
        "DENSIDAD_POB": float(ultimo["DENSIDAD_POB"]),
        "LAG_1": lag_1, "LAG_3": lag_3,
        "PROMEDIO_MOVIL_3M": promedio_movil,
        "VAR_INTERANUAL": float(ultimo["VAR_INTERANUAL"]),
        "DISTRITO": distrito, "TIPO_DELITO": tipo_delito,
        "ultimo_mes_real": (anio_ref, mes_ref),
    }


def proyectar_horizonte(spark, pipeline, modelo_clf, modelo_reg, etiquetas_riesgo,
                         hist: pd.DataFrame, distrito: str, tipo_delito: str, n_meses: int = 3):
    """Replica _proyectar_horizonte del notebook: encadena hasta n_meses de proyección.
    A diferencia de una versión simplificada, aquí cada paso EJECUTA los modelos reales
    y usa la CANTIDAD DE DELITOS ESTIMADA por el modelo (no un promedio) como insumo
    del siguiente mes — igual que en el notebook. Por eso cada mes puede dar un
    resultado distinto en vez de repetirse."""
    sub = hist[(hist.DISTRITO == distrito) & (hist.TIPO_DELITO == tipo_delito)].sort_values(["ANIO", "MES"])
    if sub.empty:
        return []

    historial = {(int(r.ANIO), int(r.MES)): float(r.CANTIDAD_DELITOS) for r in sub.itertuples()}
    meses_reales = set(historial.keys())

    ultimo = sub.iloc[-1]
    anio_actual, mes_actual = int(ultimo["ANIO"]), int(ultimo["MES"])
    var_interanual_base = float(ultimo["VAR_INTERANUAL"])
    provincia = ultimo["PROVINCIA"]

    resultados = []

    for paso in range(1, n_meses + 1):
        mes_obj, anio_obj = mes_actual + 1, anio_actual
        if mes_obj > 12:
            mes_obj, anio_obj = 1, anio_obj + 1

        lag_1 = historial.get((anio_actual, mes_actual))

        m3, a3 = mes_obj - 3, anio_obj
        while m3 <= 0:
            m3 += 12
            a3 -= 1
        lag_3 = historial.get((a3, m3))

        ventana, n_proyectados_ventana = [], 0
        for k in range(1, 4):
            mk, ak = mes_obj - k, anio_obj
            while mk <= 0:
                mk += 12
                ak -= 1
            v = historial.get((ak, mk))
            if v is not None:
                ventana.append(v)
                if (ak, mk) not in meses_reales:
                    n_proyectados_ventana += 1
        promedio_movil = sum(ventana) / len(ventana) if ventana else None

        if lag_1 is None or lag_3 is None or promedio_movil is None:
            break

        pob_distrito = hist[hist.DISTRITO == distrito]
        pob_anio_obj = pob_distrito[pob_distrito.ANIO == anio_obj]
        if not pob_anio_obj.empty:
            poblacion = float(pob_anio_obj.iloc[0]["POBLACION"])
            densidad = float(pob_anio_obj.iloc[0]["DENSIDAD_POB"])
        else:
            ultima_pob = pob_distrito.sort_values("ANIO").iloc[-1]
            poblacion = float(ultima_pob["POBLACION"])
            densidad = float(ultima_pob["DENSIDAD_POB"])

        features = {
            "POBLACION": poblacion, "DENSIDAD_POB": densidad,
            "LAG_1": lag_1, "LAG_3": lag_3,
            "PROMEDIO_MOVIL_3M": promedio_movil, "VAR_INTERANUAL": var_interanual_base,
            "DISTRITO": distrito, "TIPO_DELITO": tipo_delito,
        }
        nivel_riesgo, tasa_predicha = ejecutar_prediccion(spark, pipeline, modelo_clf, modelo_reg, etiquetas_riesgo, features)
        delitos_estimados = round(tasa_predicha * poblacion / 10_000)

        clave_confianza = min(paso - 1, 2)
        etiqueta_conf, color_conf, detalle_conf = NIVEL_CONFIANZA[clave_confianza]

        resultados.append({
            "ANIO": anio_obj, "MES": mes_obj, "PROVINCIA": provincia,
            "POBLACION": poblacion,
            "nivel_riesgo": nivel_riesgo, "tasa_predicha": tasa_predicha,
            "delitos_estimados": delitos_estimados,
            "etiqueta_conf": etiqueta_conf, "color_conf": color_conf, "detalle_conf": detalle_conf,
        })

        # Igual que en el notebook: el resultado PREDICHO por el modelo (no un promedio)
        # pasa a formar parte del "historial" para construir el vector del siguiente mes.
        historial[(anio_obj, mes_obj)] = float(delitos_estimados)
        anio_actual, mes_actual = anio_obj, mes_obj

    return resultados


# ----------------------------------------------------------------------------------
# COMPONENTES VISUALES REUTILIZABLES
# ----------------------------------------------------------------------------------
def mostrar_resultado(nivel_riesgo: str, tasa_predicha: float, poblacion: float, contexto: str = ""):
    color = COLOR_RIESGO.get(nivel_riesgo, "#7f8c8d")
    delitos_estimados = round(tasa_predicha * poblacion / 10_000)

    st.markdown(
        f"""
        <div style="padding:1.2rem;border-radius:12px;background-color:{color}22;
                    border:2px solid {color};margin-bottom:0.8rem;">
            <span style="font-size:1.1rem;font-weight:600;color:{color};">
                NIVEL DE RIESGO: {nivel_riesgo}
            </span><br>
            <span style="color:#444;">{contexto}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Nivel de riesgo", nivel_riesgo)
    c2.metric("Tasa estimada (por 10,000 hab.)", f"{tasa_predicha:.2f}")
    c3.metric("Delitos estimados", f"{delitos_estimados:,}")

    st.info(f"**Recomendación operativa:** {RECOMENDACION_RIESGO.get(nivel_riesgo, 'Sin recomendación disponible.')}")
    return delitos_estimados


def mostrar_comparacion_historica(hist: pd.DataFrame, distrito: str, tipo_delito: str,
                                   anio: int, mes: int, delitos_estimados: float):
    """Replica el bloque 'Comparación histórica' del notebook: mes anterior y
    mismo mes del año anterior, comparados contra la cantidad ESTIMADA actual."""
    reg_mes_anterior = buscar_registro_previo(hist, distrito, tipo_delito, anio, mes, meses_atras=1)
    reg_anio_anterior = buscar_registro_previo(hist, distrito, tipo_delito, anio, mes, anios_atras=1)

    cantidad_mes_anterior = float(reg_mes_anterior["CANTIDAD_DELITOS"]) if reg_mes_anterior is not None else None
    cantidad_anio_anterior = float(reg_anio_anterior["CANTIDAD_DELITOS"]) if reg_anio_anterior is not None else None

    var_mes = variacion_pct(delitos_estimados, cantidad_mes_anterior)
    var_anio = variacion_pct(delitos_estimados, cantidad_anio_anterior)

    st.markdown("### 📊 Comparación histórica")
    cc1, cc2 = st.columns(2)
    with cc1:
        if cantidad_mes_anterior is None:
            st.metric("Mes anterior", "Sin dato histórico")
        else:
            st.metric("Mes anterior", f"{cantidad_mes_anterior:,.0f} delitos",
                       delta=f"{var_mes:+.1f}%" if var_mes is not None else None,
                       delta_color="inverse")
    with cc2:
        if cantidad_anio_anterior is None:
            st.metric("Mismo mes, año anterior", "Sin dato histórico")
        else:
            st.metric("Mismo mes, año anterior", f"{cantidad_anio_anterior:,.0f} delitos",
                       delta=f"{var_anio:+.1f}%" if var_anio is not None else None,
                       delta_color="inverse")


# ----------------------------------------------------------------------------------
# PESTAÑA: RESUMEN (dashboard general, no necesita Spark — carga instantánea)
# ----------------------------------------------------------------------------------
def tab_resumen(hist: pd.DataFrame):
    st.subheader("Panorama general del dataset histórico")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros históricos", f"{len(hist):,}")
    c2.metric("Distritos", f"{hist['DISTRITO'].nunique()}")
    c3.metric("Tipos de delito", f"{hist['TIPO_DELITO'].nunique()}")
    c4.metric("Total de delitos registrados", f"{int(hist['CANTIDAD_DELITOS'].sum()):,}")

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown("**Distribución de nivel de riesgo (histórico real)**")
        conteo_riesgo = hist["NIVEL_RIESGO"].value_counts().reindex(["BAJO", "MEDIO", "ALTO"]).fillna(0)
        st.bar_chart(conteo_riesgo)

    with col_der:
        st.markdown("**Top 10 distritos por cantidad total de delitos**")
        top_distritos = hist.groupby("DISTRITO")["CANTIDAD_DELITOS"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_distritos)

    st.markdown("**Evolución mensual de delitos registrados (todos los distritos)**")
    tendencia = (
        hist.groupby(["ANIO", "MES"])["CANTIDAD_DELITOS"].sum()
        .reset_index()
        .sort_values(["ANIO", "MES"])
    )
    tendencia["PERIODO"] = tendencia["ANIO"].astype(str) + "-" + tendencia["MES"].astype(str).str.zfill(2)
    st.line_chart(tendencia.set_index("PERIODO")["CANTIDAD_DELITOS"])

    st.markdown("**Delitos por tipo (total histórico)**")
    por_tipo = hist.groupby("TIPO_DELITO")["CANTIDAD_DELITOS"].sum().sort_values(ascending=False)
    st.bar_chart(por_tipo)


# ----------------------------------------------------------------------------------
# APP PRINCIPAL
# ----------------------------------------------------------------------------------
def main():
    st.title("🚓 Predicción de Riesgo Delictivo — Región La Libertad")
    st.caption(
        "Basado en RandomForestClassifier (nivel de riesgo) y RandomForestRegressor "
        "(tasa de delitos por 10,000 hab.) entrenados con PySpark MLlib sobre datos "
        "históricos de la PNP (2022–2026)."
    )

    hist = cargar_historico()

    tab_resumen_ui, tab_predecir, tab_proyectar = st.tabs([
        "📊 Resumen", "🔍 Predecir (dato histórico)", "🔮 Proyectar (meses futuros)",
    ])

    # ============================== TAB 0: RESUMEN ==============================
    with tab_resumen_ui:
        tab_resumen(hist)

    # ============================== TAB 1: PREDECIR ==============================
    with tab_predecir:
        st.subheader("Consultar la predicción del modelo sobre un periodo ya registrado")
        st.caption(
            f"Solo se pueden consultar los años {ANIO_MIN_PREDICCION}+ (conjunto de PRUEBA del modelo). "
            "Selecciona Provincia → Distrito → Tipo de delito → Año → Mes."
        )

        hist_prueba = hist[hist.ANIO >= ANIO_MIN_PREDICCION]

        col1, col2, col3 = st.columns(3)
        with col1:
            provincia_sel = st.selectbox("Provincia", sorted(hist_prueba["PROVINCIA"].unique()), key="prov_pred")
        distritos_disp = sorted(hist_prueba.loc[hist_prueba.PROVINCIA == provincia_sel, "DISTRITO"].unique())
        with col2:
            distrito_sel = st.selectbox("Distrito", distritos_disp, key="dist_pred")
        tipos_disp = sorted(
            hist_prueba.loc[
                (hist_prueba.PROVINCIA == provincia_sel) & (hist_prueba.DISTRITO == distrito_sel), "TIPO_DELITO"
            ].unique()
        )
        with col3:
            tipo_sel = st.selectbox("Tipo de delito", tipos_disp, key="tipo_pred")

        base_filtrada = hist_prueba[
            (hist_prueba.PROVINCIA == provincia_sel) & (hist_prueba.DISTRITO == distrito_sel) &
            (hist_prueba.TIPO_DELITO == tipo_sel)
        ]
        col4, col5 = st.columns(2)
        with col4:
            anio_sel = st.selectbox("Año", sorted(base_filtrada["ANIO"].unique(), reverse=True), key="anio_pred")
        meses_disp = sorted(base_filtrada.loc[base_filtrada.ANIO == anio_sel, "MES"].unique())
        with col5:
            mes_sel = st.selectbox("Mes", meses_disp, format_func=lambda m: MESES_NOMBRE.get(m, m), key="mes_pred")

        if st.button("🔍 Predecir", type="primary", use_container_width=True):
            fila = base_filtrada[(base_filtrada.ANIO == anio_sel) & (base_filtrada.MES == mes_sel)]
            if fila.empty:
                st.warning("No se encontró un registro exacto para esa combinación.")
            else:
                fila = fila.iloc[0]
                spark, pipeline, modelo_clf, modelo_reg, etiquetas_riesgo = cargar_spark_y_modelos()
                nivel_riesgo, tasa_predicha = ejecutar_prediccion(
                    spark, pipeline, modelo_clf, modelo_reg, etiquetas_riesgo, fila.to_dict()
                )

                st.markdown("### Resultado de la predicción")
                delitos_estimados = mostrar_resultado(
                    nivel_riesgo, tasa_predicha, fila["POBLACION"],
                    contexto=f"{distrito_sel} — {tipo_sel} — {MESES_NOMBRE.get(mes_sel)} {anio_sel}",
                )

                st.markdown("### Comparación con el valor real registrado del mismo periodo")
                cc1, cc2 = st.columns(2)
                cc1.metric("Nivel de riesgo REAL", fila["NIVEL_RIESGO"])
                cc2.metric(
                    "Tasa REAL (por 10,000 hab.)", f"{fila['TASA_DELITOS_10K']:.2f}",
                    delta=f"{tasa_predicha - fila['TASA_DELITOS_10K']:+.2f} vs. predicho",
                )

                # Comparación mes anterior / mismo mes año anterior (usa historial completo, incluye 2022-2024)
                mostrar_comparacion_historica(hist, distrito_sel, tipo_sel, anio_sel, mes_sel, delitos_estimados)

                with st.expander("Ver variables utilizadas por el modelo"):
                    st.dataframe(
                        fila[[
                            "POBLACION", "DENSIDAD_POB", "LAG_1", "LAG_3",
                            "PROMEDIO_MOVIL_3M", "VAR_INTERANUAL", "DISTRITO", "TIPO_DELITO",
                        ]].to_frame().T,
                        use_container_width=True,
                    )

    # ============================== TAB 2: PROYECTAR ==============================
    with tab_proyectar:
        st.subheader("Pronosticar meses que aún no han sido registrados")
        st.caption(
            "Se proyecta el mes inmediato siguiente al último dato disponible de un distrito "
            "y tipo de delito (útil para anticipar riesgo antes de que ocurra)."
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            provincia_p = st.selectbox("Provincia", sorted(hist["PROVINCIA"].unique()), key="prov_proy")
        distritos_p = sorted(hist.loc[hist.PROVINCIA == provincia_p, "DISTRITO"].unique())
        with col2:
            distrito_p = st.selectbox("Distrito", distritos_p, key="dist_proy")
        tipos_p = sorted(hist.loc[hist.DISTRITO == distrito_p, "TIPO_DELITO"].unique())
        with col3:
            tipo_p = st.selectbox("Tipo de delito", tipos_p, key="tipo_proy")
        with col4:
            n_meses = st.slider("Meses a proyectar", min_value=1, max_value=3, value=1)

        if st.button("🔮 Proyectar", type="primary", use_container_width=True):
            spark, pipeline, modelo_clf, modelo_reg, etiquetas_riesgo = cargar_spark_y_modelos()

            if n_meses == 1:
                features_mes = calcular_features_mes_siguiente(hist, distrito_p, tipo_p)
                if features_mes is None:
                    st.warning(
                        "No hay historia suficiente (se requieren al menos 3 meses previos) "
                        "para proyectar este distrito/tipo de delito."
                    )
                else:
                    nivel_riesgo, tasa_predicha = ejecutar_prediccion(
                        spark, pipeline, modelo_clf, modelo_reg, etiquetas_riesgo, features_mes
                    )
                    st.markdown(
                        f"### Proyección para {MESES_NOMBRE.get(features_mes['MES'])} {features_mes['ANIO']}"
                    )
                    delitos_estimados = mostrar_resultado(
                        nivel_riesgo, tasa_predicha, features_mes["POBLACION"],
                        contexto=f"{distrito_p} — {tipo_p} — Confianza: Alta (dato inmediato siguiente)",
                    )
                    anio_r, mes_r = features_mes["ultimo_mes_real"]
                    mostrar_comparacion_historica(
                        hist, distrito_p, tipo_p, features_mes["ANIO"], features_mes["MES"], delitos_estimados
                    )
            else:
                pasos = proyectar_horizonte(
                    spark, pipeline, modelo_clf, modelo_reg, etiquetas_riesgo, hist, distrito_p, tipo_p, n_meses=n_meses
                )
                if not pasos:
                    st.warning(
                        "No hay historia suficiente (se requieren al menos 3 meses previos) "
                        "para proyectar este distrito/tipo de delito."
                    )
                else:
                    st.markdown(f"### Proyección a {len(pasos)} mes(es) — {distrito_p} / {tipo_p}")
                    st.caption(
                        "La confianza baja en cada paso porque los meses proyectados usan "
                        "resultados propios de pasos anteriores como si fueran datos reales."
                    )
                    for paso in pasos:
                        st.markdown(f"**{MESES_NOMBRE.get(paso['MES'])} {paso['ANIO']}**")
                        mostrar_resultado(
                            paso["nivel_riesgo"], paso["tasa_predicha"], paso["POBLACION"],
                            contexto=(
                                f"Confianza: <span style='color:{paso['color_conf']};font-weight:600;'>"
                                f"{paso['etiqueta_conf']}</span> — {paso['detalle_conf']}"
                            ),
                        )
                        st.divider()

    st.markdown("---")
    st.caption(
        "⚠️ Herramienta de apoyo a la toma de decisiones. Las predicciones deben complementarse "
        "siempre con el criterio operativo de la PNP."
    )


if __name__ == "__main__":
    main()

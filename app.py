import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Observatorio Huntington",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #F8F5FC;
}
h1, h2, h3 {
    color: #4B287C;
}
section[data-testid="stSidebar"] {
    background-color: #EEE5FA;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#59358C", "#9673C4", "#BDA7DE", "#7654A7"
]

# ==========================================
# DATOS DE DEMOSTRACIÓN
# ==========================================

@st.cache_data
def crear_datos():
    rng = np.random.default_rng(42)
    n = 400

    edad = rng.integers(18, 81, n)
    cag = rng.integers(15, 56, n)

    # Relaciones artificiales, no clínicas
    senal = cag + (edad - 40) * 0.13
    senal = senal + rng.normal(0, 6, n)

    sintomas = np.where(
        senal < 33, "Sin síntomas registrados",
        np.where(senal < 44, "Leves", "Moderados")
    )

    corea = np.clip(
        (senal - 25) / 3 + rng.normal(0, 2, n),
        0, 20
    )

    cognicion = np.clip(
        (senal - 28) / 4 + rng.normal(0, 2, n),
        0, 15
    )

    return pd.DataFrame({
        "Edad": edad,
        "CAG": cag,
        "Sintomas_motores": sintomas,
        "Puntaje_corea": np.round(corea, 1),
        "Puntaje_cognitivo": np.round(cognicion, 1)
    })


def grupo_cag(valor):
    if valor <= 26:
        return "No expandido"
    elif valor <= 35:
        return "Intermedio"
    elif valor <= 39:
        return "Penetrancia reducida"
    else:
        return "Penetrancia completa"


def preparar_datos(datos):
    datos = datos.copy()

    for columna in [
        "Edad", "CAG",
        "Puntaje_corea", "Puntaje_cognitivo"
    ]:
        datos[columna] = pd.to_numeric(
            datos[columna], errors="coerce"
        )

    datos = datos.dropna(
        subset=["Edad", "CAG"]
    )

    datos = datos[
        datos["Edad"].between(0, 110)
        & datos["CAG"].between(10, 120)
    ].copy()

    datos["Grupo_CAG"] = datos["CAG"].apply(
        grupo_cag
    )

    datos["Grupo_edad"] = pd.cut(
        datos["Edad"],
        bins=[-1, 29, 44, 59, 110],
        labels=["0-29", "30-44", "45-59", "60+"]
    )

    return datos


if "datos" not in st.session_state:
    st.session_state.datos = crear_datos()
    st.session_state.origen = "Datos sintéticos"

# ==========================================
# MENÚ LATERAL
# ==========================================

with st.sidebar:

    st.title("🧬 HUNTINGTON")
    st.subheader("Observatorio de Investigación")

    pagina = st.radio(
        "Explora la investigación",
        [
            "Inicio",
            "Genética y CAG",
            "Análisis de datos",
            "Modelo predictivo",
            "Tratamientos y conclusiones",
            "Quiz educativo"
        ]
    )

    st.divider()

    st.caption(
        "Proyecto académico de Biotecnología"
    )

    st.caption(
        "Público: investigadores y estudiantes"
    )

    archivo = st.file_uploader(
        "Importar datos de investigación (CSV)",
        type=["csv"]
    )

    if archivo is not None:
        if st.button("Cargar archivo"):
            try:
                nuevos = pd.read_csv(archivo)

                columnas = {
                    "Edad", "CAG",
                    "Sintomas_motores",
                    "Puntaje_corea",
                    "Puntaje_cognitivo"
                }

                if not columnas.issubset(nuevos.columns):
                    st.error(
                        "El CSV no tiene las columnas necesarias."
                    )
                elif nuevos.empty or len(nuevos) > 100000:
                    st.error(
                        "El archivo debe tener entre 1 y 100000 filas."
                    )
                else:
                    nuevos = preparar_datos(nuevos)

                    if nuevos.empty:
                        st.error("No hay registros válidos.")
                    else:
                        st.session_state.datos = nuevos
                        st.session_state.origen = "CSV importado"
                        st.rerun()

            except Exception as error:
                st.error(str(error))

    if st.button("Restaurar datos de ejemplo"):
        st.session_state.datos = crear_datos()
        st.session_state.origen = "Datos sintéticos"
        st.rerun()


df = preparar_datos(st.session_state.datos)

st.caption(
    "Fuente de datos: " + st.session_state.origen
)

if st.session_state.origen == "Datos sintéticos":
    st.warning(
        "Los datos son simulados. Los patrones fueron "
        "generados artificialmente y no representan "
        "resultados clínicos reales."
    )

# ==========================================
# PÁGINA 1: INICIO
# ==========================================

if pagina == "Inicio":

    st.title("🧬 Observatorio de Huntington")

    st.subheader(
        "De la genética a las manifestaciones clínicas"
    )

    st.info(
        "Pregunta de investigación: ¿Cómo se relacionan "
        "las repeticiones CAG del gen HTT con la edad "
        "y las manifestaciones clínicas de Huntington?"
    )

    st.header("¿Qué es Huntington?")

    st.write(
        "La enfermedad de Huntington es un trastorno "
        "neurodegenerativo hereditario que afecta "
        "progresivamente el sistema nervioso. "
        "Se produce por una expansión de repeticiones "
        "CAG en el gen HTT, localizado en el cromosoma 4."
    )

    st.write(
        "La enfermedad puede afectar los movimientos, "
        "la memoria, la planificación, el comportamiento "
        "y el estado de ánimo. Su presentación varía "
        "entre personas."
    )

    a, b, c = st.columns(3)

    a.metric("Gen", "HTT")
    b.metric("Cromosoma", "4")
    c.metric("Herencia", "Dominante")

    st.header("Principales manifestaciones")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Motoras")
        st.write(
            "Movimientos involuntarios o corea, "
            "alteraciones de coordinación, "
            "marcha y equilibrio."
        )

    with col2:
        st.subheader("Cognitivas")
        st.write(
            "Dificultades de atención, "
            "organización, planificación "
            "y funciones ejecutivas."
        )

    with col3:
        st.subheader("Psiquiátricas")
        st.write(
            "Depresión, irritabilidad, "
            "apatía y cambios conductuales."
        )

    st.header("Objetivo de la investigación")

    st.write(
        "Explorar relaciones entre variables "
        "genéticas y características clínicas "
        "mediante visualizaciones interactivas "
        "y un modelo de aprendizaje automático."
    )

    st.success(
        "Continúa con Genética y CAG para "
        "comprender la base de la investigación."
    )

# ==========================================
# PÁGINA 2: GENÉTICA
# ==========================================

elif pagina == "Genética y CAG":

    st.title("Genética y penetrancia")

    st.write(
        "El gen HTT contiene una región de repeticiones "
        "CAG. Su longitud ayuda a interpretar "
        "la expansión genética asociada a Huntington."
    )

    rangos = pd.DataFrame({
        "Repeticiones CAG": [
            "≤26", "27-35", "36-39", "≥40"
        ],
        "Interpretación": [
            "No expandido",
            "Intermedio",
            "Penetrancia reducida",
            "Penetrancia completa"
        ]
    })

    st.subheader("Clasificación genética")

    st.dataframe(
        rangos,
        hide_index=True,
        width="stretch"
    )

    st.write(
        "La penetrancia reducida significa que algunas "
        "personas con 36 a 39 repeticiones pueden "
        "desarrollar manifestaciones y otras no."
    )

    st.write(
        "Con 40 o más repeticiones se considera "
        "penetrancia completa durante una vida "
        "de duración habitual. Esto no permite "
        "conocer la edad exacta de aparición."
    )

    valor = st.slider(
        "Explora un número hipotético de CAG",
        10, 70, 38
    )

    st.metric(
        "Categoría genética",
        grupo_cag(valor)
    )

    st.subheader(
        "Distribución de CAG en la muestra"
    )

    conteos = (
        df["Grupo_CAG"]
        .value_counts()
        .reset_index()
    )

    conteos.columns = [
        "Grupo CAG", "Cantidad"
    ]

    fig = px.bar(
        conteos,
        x="Grupo CAG",
        y="Cantidad",
        color="Grupo CAG",
        title="Distribución de grupos genéticos"
    )

    st.plotly_chart(
        fig, width="stretch"
    )

    st.caption(
        "La gráfica muestra registros del dataset. "
        "No representa prevalencia ni penetrancia real."
    )

    st.subheader("Herencia genética")

    st.write(
        "Huntington presenta herencia autosómica "
        "dominante. Cuando un progenitor tiene "
        "una expansión causante en una de sus "
        "dos copias del gen HTT, cada hijo tiene "
        "un 50 % de probabilidad de heredarla."
    )

# ==========================================
# PÁGINA 3: ANÁLISIS DE DATOS
# ==========================================

elif pagina == "Análisis de datos":

    st.title("Análisis de características clínicas")

    st.write(
        "Esta sección explora la relación entre "
        "variables numéricas y categóricas."
    )

    st.subheader("Filtros de investigación")

    opciones = ["Todos"] + sorted(
        df["Grupo_CAG"].unique().tolist()
    )

    seleccion = st.selectbox(
        "Selecciona el grupo CAG",
        opciones
    )

    if seleccion == "Todos":
        filtrado = df.copy()
    else:
        filtrado = df[
            df["Grupo_CAG"] == seleccion
        ].copy()

    a, b, c = st.columns(3)

    a.metric("Registros", len(filtrado))

    b.metric(
        "Mediana CAG",
        round(filtrado["CAG"].median(), 1)
        if len(filtrado) else "—"
    )

    c.metric(
        "Mediana edad",
        round(filtrado["Edad"].median(), 1)
        if len(filtrado) else "—"
    )

    if filtrado.empty:
        st.info("No hay datos para este grupo.")

    else:

        st.header("1. Edad y síntomas motores")

        tabla = (
            filtrado.groupby(
                ["Grupo_edad", "Sintomas_motores"],
                observed=True
            )
            .size()
            .reset_index(name="Cantidad")
        )

        fig1 = px.bar(
            tabla,
            x="Grupo_edad",
            y="Cantidad",
            color="Sintomas_motores",
            barmode="group",
            title="Síntomas motores por edad"
        )

        st.plotly_chart(
            fig1, width="stretch"
        )

        st.write(
            "Este gráfico permite comparar "
            "la distribución de categorías motoras "
            "entre diferentes grupos de edad."
        )

        st.header("2. CAG y manifestaciones")

        variable = st.selectbox(
            "Selecciona una variable",
            [
                "Puntaje_corea",
                "Puntaje_cognitivo"
            ]
        )

        fig2 = px.scatter(
            filtrado,
            x="CAG",
            y=variable,
            color="Grupo_CAG",
            title="Relación entre CAG y " + variable,
            opacity=0.7
        )

        st.plotly_chart(
            fig2, width="stretch"
        )

        st.write(
            "El gráfico permite explorar "
            "asociaciones entre el número de "
            "repeticiones CAG y la variable seleccionada."
        )

        st.header("3. Comparación por grupo CAG")

        resumen = (
            filtrado.groupby("Grupo_CAG")[
                ["Puntaje_corea", "Puntaje_cognitivo"]
            ]
            .mean()
            .reset_index()
        )

        fig3 = px.bar(
            resumen,
            x="Grupo_CAG",
            y=[
                "Puntaje_corea",
                "Puntaje_cognitivo"
            ],
            barmode="group",
            title="Promedio de puntuaciones por CAG"
        )

        st.plotly_chart(
            fig3, width="stretch"
        )

        st.warning(
            "Las puntuaciones sintéticas no son "
            "escalas clínicas validadas. "
            "Una asociación no demuestra causalidad."
        )

# ==========================================
# PÁGINA 4: MODELO PREDICTIVO
# ==========================================

elif pagina == "Modelo predictivo":

    st.title("Modelo predictivo de investigación")

    st.write(
        "El objetivo es estudiar si edad y CAG "
        "permiten clasificar las categorías "
        "de síntomas motores registradas."
    )

    st.subheader("¿Cómo funciona Random Forest?")

    st.write(
        "Random Forest combina múltiples árboles "
        "de decisión. Cada árbol genera una "
        "predicción y el conjunto selecciona "
        "la clase más votada."
    )

    st.write(
        "Se compara con un modelo básico que "
        "siempre predice la categoría más frecuente."
    )

    st.info(
        "75 % de los registros se utilizan para "
        "entrenamiento y 25 % para evaluación."
    )

    datos_modelo = df.dropna(
        subset=[
            "Edad", "CAG", "Sintomas_motores"
        ]
    ).copy()

    y = datos_modelo[
        "Sintomas_motores"
    ].astype(str)

    if (
        len(datos_modelo) < 40
        or y.nunique() < 2
        or y.value_counts().min() < 4
    ):
        st.warning(
            "No hay suficientes registros "
            "para entrenar el modelo."
        )

    elif st.button(
        "Entrenar modelo",
        type="primary"
    ):

        X = datos_modelo[
            ["Edad", "CAG"]
        ]

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X, y,
                test_size=0.25,
                random_state=42,
                stratify=y
            )
        )

        modelos = {
            "Modelo base": DummyClassifier(
                strategy="most_frequent"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                random_state=42
            )
        }

        resultados = []

        for nombre, modelo in modelos.items():

            modelo.fit(X_train, y_train)
            pred = modelo.predict(X_test)

            resultados.append({
                "Modelo": nombre,
                "Exactitud": accuracy_score(
                    y_test, pred
                ),
                "F1 macro": f1_score(
                    y_test, pred,
                    average="macro",
                    zero_division=0
                )
            })

            if nombre == "Random Forest":

                etiquetas = sorted(y.unique())

                matriz = confusion_matrix(
                    y_test, pred,
                    labels=etiquetas
                )

                fig_matriz = px.imshow(
                    matriz,
                    x=etiquetas,
                    y=etiquetas,
                    text_auto=True,
                    color_continuous_scale="Purples",
                    title="Matriz de confusión"
                )

                st.plotly_chart(
                    fig_matriz,
                    width="stretch"
                )

        resultados = pd.DataFrame(
            resultados
        )

        st.subheader("Resultados")

        st.dataframe(
            resultados.round(3),
            hide_index=True,
            width="stretch"
        )

        fig = px.bar(
            resultados,
            x="Modelo",
            y=["Exactitud", "F1 macro"],
            barmode="group",
            range_y=[0, 1],
            title="Comparación de modelos"
        )

        st.plotly_chart(
            fig, width="stretch"
        )

        st.write(
            "La exactitud representa la proporción "
            "de predicciones correctas. "
            "F1 macro resume el rendimiento "
            "promedio entre categorías."
        )

        st.warning(
            "Los resultados con datos sintéticos "
            "no demuestran capacidad predictiva "
            "en pacientes reales. "
            "Se requiere validación clínica externa."
        )

# ==========================================
# PÁGINA 5: TRATAMIENTOS
# ==========================================

elif pagina == "Tratamientos y conclusiones":

    st.title("Tratamientos y conclusiones")

    st.header("Tratamientos según manifestaciones")

    st.write(
        "Actualmente, el tratamiento de Huntington "
        "se orienta principalmente al control "
        "de síntomas y al apoyo funcional."
    )

    tratamientos = pd.DataFrame({
        "Manifestación": [
            "Corea",
            "Depresión",
            "Dificultades motoras",
            "Cambios cognitivos"
        ],
        "Abordaje": [
            "Tetrabenazina o deutetrabenazina bajo supervisión especializada",
            "Evaluación de salud mental y tratamiento individualizado",
            "Fisioterapia y terapia ocupacional",
            "Evaluación neuropsicológica y apoyo funcional"
        ]
    })

    st.dataframe(
        tratamientos,
        hide_index=True,
        width="stretch"
    )

    st.warning(
        "Los medicamentos no se seleccionan "
        "únicamente por las repeticiones CAG. "
        "Esta plataforma no recomienda tratamientos "
        "para pacientes individuales."
    )

    st.header("Conclusiones de investigación")

    st.markdown("""
    - Las repeticiones CAG permiten interpretar
      categorías genéticas relacionadas con HTT.
    - La edad y los síntomas son variables
      relevantes para la investigación.
    - Los gráficos ayudan a explorar asociaciones.
    - El modelo predictivo es exploratorio.
    - Se necesitan datos clínicos reales
      para validar cualquier conclusión.
    """)

    st.header("Limitaciones")

    st.write(
        "Los datos iniciales son sintéticos. "
        "La edad registrada no representa "
        "necesariamente la edad de inicio. "
        "Las puntuaciones artificiales no "
        "son escalas clínicas validadas."
    )

    st.header("Fuentes científicas")

    st.markdown(
        "[MedlinePlus Genetics]"
        "(https://medlineplus.gov/genetics/"
        "condition/huntington-disease/)"
    )

    st.markdown(
        "[GeneReviews]"
        "(https://www.ncbi.nlm.nih.gov/books/NBK1305/)"
    )

# ==========================================
# PÁGINA 6: QUIZ
# ==========================================

elif pagina == "Quiz educativo antiguo":

    st.title("🧠 Quiz sobre Huntington")

    st.write(
        "Responde cinco preguntas para comprobar "
        "lo aprendido durante la investigación."
    )

    preguntas = [
        {
            "pregunta": "¿Qué gen se relaciona con Huntington?",
            "opciones": ["BRCA1", "HTT", "CFTR", "HBB"],
            "correcta": "HTT",
            "explicacion": "La expansión CAG causante se encuentra en el gen HTT."
        },
        {
            "pregunta": "¿Qué significa tener 36 a 39 repeticiones CAG?",
            "opciones": [
                "Penetrancia completa",
                "Penetrancia reducida",
                "Rango no expandido",
                "Ausencia de variación genética"
            ],
            "correcta": "Penetrancia reducida",
            "explicacion": "Algunas personas con 36 a 39 repeticiones desarrollan manifestaciones y otras no."
        },
        {
            "pregunta": "¿Cuál es un síntoma motor de Huntington?",
            "opciones": [
                "Anemia",
                "Hipertensión",
                "Corea",
                "Pérdida de audición"
            ],
            "correcta": "Corea",
            "explicacion": "La corea consiste en movimientos involuntarios."
        },
        {
            "pregunta": "¿Qué probabilidad tiene cada hijo de heredar una expansión causante si un progenitor es heterocigoto?",
            "opciones": [
                "25 %",
                "50 %",
                "75 %",
                "100 %"
            ],
            "correcta": "50 %",
            "explicacion": "Cada hijo tiene un 50 % de probabilidad de heredar la expansión."
        },
        {
            "pregunta": "¿Qué NO permite conocer exactamente el número CAG?",
            "opciones": [
                "La categoría genética",
                "El intervalo de repeticiones",
                "La edad exacta de aparición",
                "Si pertenece al rango 36-39"
            ],
            "correcta": "La edad exacta de aparición",
            "explicacion": "Las repeticiones CAG no determinan la edad exacta de inicio."
        }
    ]

    with st.form("quiz_huntington"):

        respuestas = []

        for i, pregunta in enumerate(preguntas):

            respuesta = st.radio(
                f"{i+1}. {pregunta['pregunta']}",
                ["Selecciona una respuesta"]
                + pregunta["opciones"],
                key=f"pregunta_{i}"
            )

            respuestas.append(respuesta)

        enviar = st.form_submit_button(
            "Ver resultados",
            type="primary"
        )

    if enviar:

        if "Selecciona una respuesta" in respuestas:
            st.warning(
                "Debes responder las cinco preguntas."
            )

        else:

            puntuacion = sum(
                respuesta == pregunta["correcta"]
                for respuesta, pregunta
                in zip(respuestas, preguntas)
            )

            st.header("Resultados")

            st.metric(
                "Calificación",
                f"{puntuacion} / 5"
            )

            st.progress(puntuacion / 5)

            if puntuacion == 5:
                st.success(
                    "¡Excelente! Dominas los conceptos."
                )

            elif puntuacion >= 3:
                st.info(
                    "¡Buen trabajo! Sigue aprendiendo."
                )

            else:
                st.warning(
                    "Te recomendamos repasar "
                    "las secciones anteriores."
                )

            st.subheader("Revisión de respuestas")

            for i, (respuesta, pregunta) in enumerate(
                zip(respuestas, preguntas)
            ):

                if respuesta == pregunta["correcta"]:
                    st.success(
                        f"Pregunta {i+1}: correcta"
                    )
                else:
                    st.error(
                        f"Pregunta {i+1}: incorrecta"
                    )

                st.write(
                    "Respuesta correcta:",
                    pregunta["correcta"]
                )

                st.caption(
                    pregunta["explicacion"]
                )

# ==========================================================
# NUEVO MÓDULO: QUIZ EDUCATIVO POR NIVELES
# ==========================================================

if pagina == "Quiz educativo":

    st.title("Quiz educativo sobre la enfermedad de Huntington")

    st.write(
        "Evalúa tus conocimientos sobre genética, "
        "manifestaciones clínicas y biología molecular."
    )

    st.info(
        "Selecciona un nivel de dificultad. Cada nivel "
        "contiene seis preguntas de selección única."
    )

    # BANCO DE PREGUNTAS
    banco_preguntas = {

        "Fácil": [
            {
                "pregunta": "1. ¿Qué gen está relacionado con la enfermedad de Huntington?",
                "opciones": ["BRCA1", "HTT", "CFTR", "HBB"],
                "correcta": "HTT",
                "explicacion": "La enfermedad se relaciona con una expansión de repeticiones CAG en el gen HTT."
            },
            {
                "pregunta": "2. ¿En qué cromosoma se encuentra el gen HTT?",
                "opciones": ["Cromosoma 1", "Cromosoma 4", "Cromosoma 7", "Cromosoma 21"],
                "correcta": "Cromosoma 4",
                "explicacion": "El gen HTT se localiza en el cromosoma 4."
            },
            {
                "pregunta": "3. ¿Qué tipo de herencia presenta Huntington?",
                "opciones": ["Autosómica dominante", "Autosómica recesiva", "Ligada al cromosoma X", "Mitocondrial"],
                "correcta": "Autosómica dominante",
                "explicacion": "Huntington presenta un patrón de herencia autosómico dominante."
            },
            {
                "pregunta": "4. ¿Qué significa CAG?",
                "opciones": ["Una proteína", "Una secuencia de tres nucleótidos", "Un cromosoma", "Una enzima"],
                "correcta": "Una secuencia de tres nucleótidos",
                "explicacion": "CAG es un triplete de nucleótidos formado por citosina, adenina y guanina."
            },
            {
                "pregunta": "5. ¿Cuál es una manifestación motora característica?",
                "opciones": ["Corea", "Anemia", "Hipertensión", "Pérdida de audición"],
                "correcta": "Corea",
                "explicacion": "La corea consiste en movimientos involuntarios que pueden presentarse en Huntington."
            },
            {
                "pregunta": "6. ¿Qué sistema del organismo se afecta progresivamente?",
                "opciones": ["Sistema digestivo", "Sistema nervioso", "Sistema respiratorio", "Sistema endocrino"],
                "correcta": "Sistema nervioso",
                "explicacion": "Huntington es una enfermedad neurodegenerativa que afecta progresivamente el sistema nervioso."
            }
        ],

        "Medio": [
            {
                "pregunta": "1. ¿Qué interpretación corresponde a 36-39 repeticiones CAG?",
                "opciones": ["No expandido", "Penetrancia reducida", "Penetrancia completa", "Ausencia de mutación"],
                "correcta": "Penetrancia reducida",
                "explicacion": "Este intervalo se relaciona con penetrancia reducida, por lo que no todas las personas desarrollan la enfermedad."
            },
            {
                "pregunta": "2. Si un progenitor es heterocigoto para la expansión causante, ¿cuál es la probabilidad de transmitirla a cada hijo?",
                "opciones": ["25 %", "50 %", "75 %", "100 %"],
                "correcta": "50 %",
                "explicacion": "Cada hijo tiene una probabilidad independiente del 50 % de heredar la expansión."
            },
            {
                "pregunta": "3. ¿Qué sucede cuando aumenta el número de repeticiones CAG?",
                "opciones": ["Se modifica la longitud de la región repetitiva", "Desaparece el gen HTT", "Se elimina el cromosoma 4", "Se detiene la expresión genética"],
                "correcta": "Se modifica la longitud de la región repetitiva",
                "explicacion": "La expansión incrementa el número de tripletes CAG presentes en el gen HTT."
            },
            {
                "pregunta": "4. ¿Qué función cumple una prueba genética para Huntington?",
                "opciones": ["Medir la presión arterial", "Analizar el número de repeticiones CAG", "Medir la actividad muscular", "Determinar el grupo sanguíneo"],
                "correcta": "Analizar el número de repeticiones CAG",
                "explicacion": "El análisis molecular permite determinar el número de repeticiones CAG en el gen HTT."
            },
            {
                "pregunta": "5. ¿Cuál es una manifestación cognitiva de Huntington?",
                "opciones": ["Dificultad de planificación", "Aumento de audición", "Mejor coordinación", "Incremento de fuerza muscular"],
                "correcta": "Dificultad de planificación",
                "explicacion": "Pueden presentarse dificultades en funciones ejecutivas, atención y planificación."
            },
            {
                "pregunta": "6. ¿Qué representa la penetrancia genética?",
                "opciones": ["La frecuencia de una proteína", "La proporción de individuos con una variante genética que manifiestan el fenotipo asociado", "La cantidad de cromosomas", "La velocidad de división celular"],
                "correcta": "La proporción de individuos con una variante genética que manifiestan el fenotipo asociado",
                "explicacion": "La penetrancia describe la proporción de individuos con un genotipo determinado que presentan el fenotipo correspondiente."
            }
        ],

        "Difícil": [
            {
                "pregunta": "1. ¿Qué aminoácido se codifica mediante el triplete CAG?",
                "opciones": ["Glicina", "Alanina", "Glutamina", "Leucina"],
                "correcta": "Glutamina",
                "explicacion": "El codón CAG codifica el aminoácido glutamina. La expansión produce una región de poliglutamina en la proteína huntingtina."
            },
            {
                "pregunta": "2. ¿Qué consecuencia molecular produce la expansión CAG en la proteína huntingtina?",
                "opciones": ["Disminución de aminoácidos", "Expansión de la región de poliglutamina", "Eliminación del ADN mitocondrial", "Formación de hemoglobina"],
                "correcta": "Expansión de la región de poliglutamina",
                "explicacion": "La expansión CAG aumenta la longitud de la región de poliglutamina de la proteína huntingtina."
            },
            {
                "pregunta": "3. ¿Qué técnica molecular puede utilizarse para determinar el número de repeticiones CAG?",
                "opciones": ["PCR", "ELISA", "Western blot", "Microscopía óptica"],
                "correcta": "PCR",
                "explicacion": "La PCR permite amplificar la región del gen HTT que contiene las repeticiones para analizar su longitud."
            },
            {
                "pregunta": "4. ¿Qué fenómeno genético explica que la enfermedad pueda aparecer a edades más tempranas en generaciones sucesivas?",
                "opciones": ["Codominancia", "Anticipación genética", "Recombinación mitocondrial", "Silenciamiento génico"],
                "correcta": "Anticipación genética",
                "explicacion": "La anticipación puede relacionarse con el aumento de repeticiones CAG entre generaciones, especialmente por inestabilidad durante la transmisión."
            },
            {
                "pregunta": "5. Un investigador observa que dos personas tienen diferente edad de inicio pese a presentar expansiones CAG similares. ¿Cuál es la interpretación más adecuada?",
                "opciones": ["La edad de inicio depende exclusivamente de CAG", "Otros factores genéticos y ambientales pueden influir", "Una de las personas necesariamente tiene otro cromosoma", "El número CAG no tiene relación con Huntington"],
                "correcta": "Otros factores genéticos y ambientales pueden influir",
                "explicacion": "La edad de inicio puede estar influenciada por factores modificadores genéticos y ambientales, además del número de repeticiones."
            },
            {
                "pregunta": "6. Un modelo Random Forest obtiene una exactitud del 90 % usando datos sintéticos. ¿Cuál es la conclusión científica más adecuada?",
                "opciones": ["El modelo diagnostica Huntington con 90 % de precisión", "El resultado demuestra eficacia clínica", "El resultado es exploratorio y requiere validación con datos reales", "El modelo sustituye las pruebas genéticas"],
                "correcta": "El resultado es exploratorio y requiere validación con datos reales",
                "explicacion": "Los datos sintéticos no demuestran capacidad predictiva en pacientes. Se necesita validación externa con datos clínicos reales."
            }
        ]
    }

    # SELECCIÓN DEL NIVEL
    nivel = st.selectbox(
        "Selecciona tu nivel de dificultad:",
        ["Fácil", "Medio", "Difícil"],
        key="nivel_quiz_huntington"
    )

    descripciones = {
        "Fácil": "Conceptos fundamentales de Huntington y genética.",
        "Medio": "Herencia, penetrancia y manifestaciones clínicas.",
        "Difícil": "Genética molecular, técnicas y análisis científico."
    }

    st.caption(descripciones[nivel])

    st.divider()

    preguntas_actuales = banco_preguntas[nivel]

    # IDENTIFICAR EL EXAMEN ACTUAL
    clave_resultado = "resultado_quiz_" + nivel
    clave_respuestas = "respuestas_quiz_" + nivel

    # FORMULARIO DEL QUIZ
    if clave_resultado not in st.session_state:

        with st.form("formulario_quiz_" + nivel):

            st.subheader("Evaluación: nivel " + nivel)

            respuestas = []

            for i, pregunta in enumerate(preguntas_actuales):

                st.markdown("**Pregunta " + str(i + 1) + " de 6**")

                respuesta = st.radio(
                    pregunta["pregunta"],
                    pregunta["opciones"],
                    index=None,
                    key="respuesta_" + nivel + "_" + str(i)
                )

                respuestas.append(respuesta)

                st.divider()

            enviar = st.form_submit_button(
                "Finalizar evaluación",
                type="primary",
                use_container_width=True
            )

        if enviar:

            if any(r is None for r in respuestas):

                st.warning(
                    "Debes responder las seis preguntas antes de finalizar."
                )

            else:

                puntuacion = sum(
                    respuesta == pregunta["correcta"]
                    for respuesta, pregunta
                    in zip(respuestas, preguntas_actuales)
                )

                st.session_state[clave_resultado] = puntuacion
                st.session_state[clave_respuestas] = respuestas

                st.rerun()

    # RESULTADOS
    else:

        puntuacion = st.session_state[clave_resultado]
        respuestas_guardadas = st.session_state[clave_respuestas]

        porcentaje = (puntuacion / 6) * 100

        st.subheader("Resultados de la evaluación")

        col1, col2, col3 = st.columns(3)

        col1.metric("Puntuación", str(puntuacion) + " / 6")
        col2.metric("Rendimiento", str(round(porcentaje)) + " %")
        col3.metric("Nivel", nivel)

        st.progress(puntuacion / 6)

        if puntuacion == 6:
            st.success("¡Excelente! Respondiste correctamente todas las preguntas.")

        elif puntuacion >= 4:
            st.info("¡Buen trabajo! Tienes un buen dominio de los conceptos.")

        else:
            st.warning("Te recomendamos repasar el contenido y volver a intentarlo.")

        st.divider()

        st.subheader("Revisión de respuestas")

        for i, (pregunta, respuesta) in enumerate(
            zip(preguntas_actuales, respuestas_guardadas)
        ):

            st.markdown("**" + pregunta["pregunta"] + "**")

            if respuesta == pregunta["correcta"]:
                st.success("Respuesta seleccionada: " + respuesta)
            else:
                st.error("Respuesta seleccionada: " + respuesta)

                st.info(
                    "Respuesta correcta: " + pregunta["correcta"]
                )

            st.caption(pregunta["explicacion"])

            st.divider()

        # REPETIR EVALUACIÓN
        if st.button(
            "Repetir este nivel",
            type="primary",
            use_container_width=True
        ):

            del st.session_state[clave_resultado]
            del st.session_state[clave_respuestas]

            for i in range(6):
                key = "respuesta_" + nivel + "_" + str(i)

                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

    st.divider()

    st.caption(
        "Evaluación educativa de apoyo académico. "
        "Los resultados no constituyen una evaluación clínica."
    )

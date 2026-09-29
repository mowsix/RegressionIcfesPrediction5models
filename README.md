# Predicción del desempeño en las Pruebas Saber 11 (ICFES)

Proyecto integrador de analítica predictiva desarrollado con la metodología **CRISP-DM**.
Estima el puntaje por área de las Pruebas Saber 11 a partir del contexto socioeconómico y
escolar del estudiante, y despliega los modelos en una aplicación web.

**Tipo de problema:** regresión · **Fuente:** [Datos Abiertos Colombia — ICFES](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe)

---

## 1. Entendimiento del negocio

Se predice el puntaje (0–100) que obtendría un estudiante en cada una de las cinco áreas
(Lectura Crítica, Matemáticas, Ciencias Naturales, Sociales y Ciudadanas e Inglés), usando
únicamente variables disponibles **antes** de presentar la prueba. Se entrenan **cinco modelos
de regresión independientes**, uno por área, con un **esquema de entrada común** para que en el
despliegue se ejecuten simultáneamente sobre un mismo formulario.

## 2. Entendimiento de los datos

- Datos extraídos vía **API SODA (Socrata)** de Datos Abiertos, consolidando los periodos 2018-1 y 2019-1.
- **Sesgo documentado:** al ser aplicaciones de primer semestre (predominio de Calendario B, en su
  mayoría colegios no oficiales), la muestra generaliza mejor a esa población.
- **Nota de extracción:** los periodos de mayor volumen fallaban con HTTP 400 por un desajuste de tipos
  en la columna de paginación y por la paginación profunda; se consolidó sobre los periodos extraídos.

## 3. Preparación de los datos

- Descarte de **fugas de datos** (todos los `punt_*`, `percentil_*`, `desemp_*` salvo el objetivo de cada modelo),
  **identificadores** y **códigos redundantes**.
- Limpieza de errores: normalización de texto, unificación de tokens de nulo (`SIN DATO`, `N/A`, …),
  eliminación de duplicados y de puntajes fuera de rango.
- **Nulos:** columnas con >20% de nulos eliminadas; categóricas imputadas con la categoría explícita
  `SIN_DATO` (sin *leakage*); objetivos nunca imputados.
- **Descarte adicional** de variables no informativas (`periodo`, `estu_tipodocumento`, `cole_calendario`),
  columnas casi constantes (>98% una categoría) y geográficas redundantes.
- **Selección por asociación:** relevancia con **η² (correlation ratio)** frente a los puntajes y
  redundancia entre predictoras con **Cramér's V**. Resultado: se descarta lo irrelevante
  (p. ej. `estu_genero`) y se conservan **16 variables predictoras**.
- **Codificación:** One-Hot Encoding (`handle_unknown='ignore'`) dentro de un `Pipeline`/`ColumnTransformer`.
- EDA automático con **ydata-profiling** (reporte HTML).

## 4. Modelamiento

- **Modelos clásicos:** Regresión Lineal, Ridge, SVM (LinearSVR), KNN y Red Neuronal (MLP).
- **Ensambles:** Bagging (Random Forest), Boosting (HistGradientBoosting) y Voting.
- **Ajuste de hiperparámetros** de cada modelo con `RandomizedSearchCV` y validación cruzada (3-fold)
  sobre el 70% de entrenamiento; grillas amplias por modelo.
- Selección **automática del mejor modelo por área** según RMSE en validación cruzada, reentrenado
  luego sobre todo el conjunto de entrenamiento.

## 5. Evaluación

- Métrica principal **RMSE** (en unidades del puntaje, penaliza errores grandes), complementada con
  **MAE** y **R²**, medidas sobre el conjunto de prueba (30%).
- Se selecciona el mejor modelo por área justificando la decisión con las métricas.

## 6. Despliegue

Aplicación **Streamlit** que carga los cinco modelos serializados y el esquema de entrada, presenta
un formulario con las variables de contexto y devuelve las cinco predicciones más un puntaje global
estimado (ponderación oficial del ICFES).

---

## Estructura del repositorio

```
├── app.py                              # Aplicación Streamlit (despliegue)
├── requirements.txt                    # Dependencias (scikit-learn fijado a la versión de entrenamiento)
├── esquema_entrada.joblib              # Columnas y categorías válidas de entrada
├── modelo_punt_lectura_critica.joblib  # Un modelo por área
├── modelo_punt_matematicas.joblib
├── modelo_punt_c_naturales.joblib
├── modelo_punt_sociales_ciudadanas.joblib
└── modelo_punt_ingles.joblib
```

Los cuadernos de **preparación** y **minería** (Google Colab) contienen el flujo CRISP-DM completo.

## Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```

Despliegue en la nube: subir el repositorio a GitHub y publicarlo desde
[Streamlit Community Cloud](https://share.streamlit.io) indicando `app.py` como archivo principal.

---

> **Nota metodológica.** Las predicciones son estimaciones estadísticas basadas en variables de
> contexto, no una medición del desempeño real del estudiante. Los modelos se entrenaron con los
> periodos 2018-1 y 2019-1 (muestra sesgada hacia colegios de Calendario B / no oficiales).

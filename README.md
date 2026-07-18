# SpaML_M003 — Predicción de ocupación de un spa

Proyecto de Machine Learning (Project Break II — bootcamp Data Science & IA) sobre la
ocupación real de un spa en Sevilla: cuántas citas se van a ejecutar en cada tramo
horario, para apoyar la planificación de personal y cabinas.

[Español](#español) | [English](#english)

---

## Español

### Descripción del problema

El spa organiza su operativa diaria en dos **tramos horarios** (mañana, hasta las
14:00, y tarde, desde las 14:00). Hoy esa planificación de personal y cabinas se
decide por intuición o por la media histórica, sin anticipar picos ni valles de
demanda.

**Objetivo:** predecir el número de citas (`n_citas`) que se ejecutarán en cada
tramo horario de cada día, para que el negocio pueda:

- Evitar sobre-dotación de personal en tramos flojos (coste).
- Evitar infra-dotación en tramos punta (pérdida de reservas o mala experiencia).
- Anticipar la demanda de fechas atípicas (festivos, San Valentín, temporada alta).

El problema, la hipótesis de modelado y su justificación completa están documentados
en [`src/docs/business_case.md`](src/docs/business_case.md).

### Dataset utilizado

**Datos privados y reales** de un negocio en funcionamiento (no un dataset público de
práctica): exportes del sistema de reservas del spa, con una fila por línea de
pago/reserva.

- **Export crudo** (`informe_NOUP.csv`): contiene datos personales de clientes reales
  (nombre, teléfono, email) y **no se sube al repositorio** por protección de datos
  (RGPD) — está excluido explícitamente en `.gitignore` (patrón `*_NOUP*`). Solo
  quien tenga acceso al sistema de reservas del negocio puede regenerarlo.
- **Datos derivados versionados** (`src/data_sample/processed/`): agregados **sin
  ninguna información personal** — solo fecha, tramo horario y variables de
  calendario. Son el resultado de `src/notebooks/transform.ipynb`, que descarta
  cualquier columna identificativa nada más leer el crudo.
- **Cobertura:** 9.081 líneas de pago → 8.106 reservas únicas → 6.190 reservas de
  servicio (se excluyen tarjetas regalo y membresías, que no son citas) → 6.101
  confirmadas (89 canceladas) → 6.040 tras el corte temporal al 30-jun-2026.
- **Dataset final:** 1.566 filas (fecha × tramo), del 2024-05-09 al 2026-06-30, sin
  huecos relevantes.
- **Split train/test cronológico** (antes de cualquier EDA, para evitar fuga
  temporal): train hasta 2026-01-24 (1.252 filas, 626 días), test desde 2026-01-25
  hasta 2026-06-30 (314 filas, 157 días).

### Solución adoptada

**Regresión supervisada sobre serie temporal.** Target: `n_citas` por fecha × tramo
horario (un conteo entero ≥ 0).

1. **Transformación** (`transform.ipynb`): del ticket individual a la rejilla
   fecha × tramo, con split train/test cronológico.
2. **EDA dirigido al modelado** (`eda.ipynb`): estructura semanal marcada (no
   gradual), la tarde casi dobla a la mañana, estacionalidad anual clara y una
   tendencia de crecimiento real del negocio.
3. **Feature engineering** (`feature_engineering.ipynb`): variables de tendencia,
   agrupación de días por comportamiento (`grupo_dia`), temporada, festivos
   (librería `holidays`, calendario de Andalucía) y fechas comerciales — validadas
   contra el target antes de incorporarlas, no por intuición.
4. **Preprocesado** (`feature_preprocessing.ipynb`): exclusión de días de cierre del
   negocio, one-hot encoding de variables categóricas y escalado de la variable de
   tendencia (`StandardScaler`, ajustado solo en train).
5. **Modelado** (`modeling.ipynb`): métrica MAE (el MAPE es inviable — un 21% de las
   mañanas tienen 0 citas), dos baselines, comparativa de 6 algoritmos con
   validación cruzada temporal (`TimeSeriesSplit`, nunca `KFold` aleatorio), y
   optimización de hiperparámetros (`RandomizedSearchCV`) de los dos mejores.
6. **Evaluación final** (`evaluation.ipynb`): una única evaluación contra test,
   análisis de residuos, error por segmento e interpretabilidad.

Todo el pipeline está también ensamblado en [`main.ipynb`](main.ipynb) (notebook
único, ejecutable de principio a fin).

### Estructura del repositorio

```
├── src/
│   ├── data_sample/
│   │   └── processed/     # Datos derivados versionados (agregados, sin PII)
│   ├── docs/
│   │   └── business_case.md  # Problema de negocio, hipótesis y justificación
│   ├── img/               # Recursos gráficos
│   ├── models/            # Modelo y scaler entrenados (joblib)
│   ├── notebooks/         # Pipeline de desarrollo, un notebook por fase
│   │   ├── transform.ipynb
│   │   ├── eda.ipynb
│   │   ├── feature_engineering.ipynb
│   │   ├── feature_preprocessing.ipynb
│   │   ├── modeling.ipynb
│   │   └── evaluation.ipynb
│   └── utils/             # Funciones reutilizadas por los notebooks
├── main.ipynb             # Pipeline completo ensamblado, de principio a fin
├── Presentación.pdf       # Presentación del proyecto
├── requirements.txt
└── README.md
```

### Tecnologías utilizadas

- **Python** (pandas, numpy)
- **scikit-learn** — modelado, validación cruzada temporal, búsqueda de
  hiperparámetros
- **matplotlib, seaborn** — visualización
- **holidays** — calendario oficial de festivos (Andalucía)
- **joblib** — persistencia del modelo y del scaler
- **Jupyter Notebook**
- **Git / GitHub** — Git Flow con ramas `feature/*`, Pull Requests revisadas e
  issues para el seguimiento del trabajo pendiente

### Instrucciones de reproducción

```bash
git clone https://github.com/Emigarsan/ML_Spa_M003.git
cd ML_Spa_M003
pip install -r requirements.txt
```

**Para ejecutar `main.ipynb` de principio a fin** hace falta una copia local propia
de `informe_NOUP.csv` (el export confidencial, ver más arriba) en la raíz del
repositorio — sin él, el PRIMER PASO falla. El resto de fases (a partir de
`src/data_sample/processed/`) no lo necesitan porque parten de datos ya derivados
y versionados.

**Para usar directamente el modelo ya entrenado** (sin reejecutar todo el pipeline):

```python
import joblib
art = joblib.load('src/models/modelo_ocupacion.joblib')
modelo, columnas = art['modelo'], art['columnas']
# modelo.predict(X) sobre un DataFrame con las mismas `columnas`,
# tras aplicar src/utils/preprocessing.build_features() y el scaler
# guardado en src/models/scaler.joblib
```

### Principales resultados

**Métrica principal: MAE** (error medio en nº de citas por tramo — la unidad con la
que el negocio decide personal en sala). El MAPE se descartó por inviable.

| Modelo | MAE (CV temporal, train) |
|---|---|
| Baseline media | 2,30 |
| Baseline estacional (t-7) | 2,02 |
| Comparativa de 6 modelos, mejor (GradientBoosting) | 1,88 |
| **RandomForest optimizado (RandomizedSearchCV)** | **1,74** |

**Evaluación final, única vez contra test** (2026-01-25 → 2026-06-30, nunca visto
antes):

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Baseline media de train | 2,76 | 3,70 | −0,22 |
| Baseline estacional (t-7) | 2,46 | 3,16 | 0,11 |
| **RandomForest (final)** | **1,92** | **2,52** | **0,43** |

El MAE de test (1,92) es consistente con el estimado por validación cruzada (1,74):
la metodología temporal fue honesta, sin sobreajuste severo.

**Interpretabilidad** (importancia de features del modelo final): la tendencia
temporal pesa más (`dias_desde_inicio`, 0,35), seguida del tramo (`tramo_tarde`,
0,21) y el día de la semana (0,19) — coherente con las conclusiones del EDA.

**Lectura de negocio:** un MAE de ~2 citas sobre una demanda media de ~5 citas/tramo
permite dimensionar personal por franjas (flojo / normal / punta), pero no asignar
cabina a cabina. El error se concentra en tarde y fin de semana, justo donde más
demanda hay — el negocio debería tratar la predicción como un suelo, no como un
techo, al dotar personal.

**Limitaciones y próximas mejoras** (ninguna es "probar otro modelo"):
1. Incorporar *lags* (t-7, t-14) con protocolo de predicción rolling, si el negocio
   confirma un horizonte de predicción ≤ 7 días (correlación ~0,55 ya medida).
2. Calendario oficial de cierres del negocio como dato de entrada, en vez de
   inferirlo de la demanda cero.
3. Reevaluar con un tercer ciclo anual completo de datos.
4. Predicción por intervalos (cuantiles) en vez de una cifra única.

### Autores

- **Emilio Garrote** — [github.com/Emigarsan](https://github.com/Emigarsan)
- **lolarealcejudo27** — [github.com/lolarealcejudo27](https://github.com/lolarealcejudo27)
- **MCCFern** — [github.com/MCCFern](https://github.com/MCCFern)

---

## English

### Problem description

The spa organizes its daily operations in two **time slots** (morning, until 14:00,
and afternoon, from 14:00). Staff and room planning is currently decided by
intuition or historical averages, without anticipating demand peaks or troughs.

**Goal:** predict the number of appointments (`n_citas`) per time slot and day, so
the business can avoid both over- and under-staffing, and anticipate demand on
atypical dates (holidays, Valentine's Day, high season).

Full business case and modeling hypothesis: [`src/docs/business_case.md`](src/docs/business_case.md).

### Dataset

**Private, real data** from an operating business (not a public practice dataset):
exports from the spa's booking system.

- The **raw export** (`informe_NOUP.csv`) contains real customers' personal data
  (name, phone, email) and is **excluded from the repository** (GDPR) via
  `.gitignore` (`*_NOUP*` pattern).
- **Versioned derived data** (`src/data_sample/processed/`) are aggregates **with
  no personal information** — only date, time slot and calendar variables.
- Pipeline: 9,081 payment lines → 8,106 unique bookings → 6,190 service bookings →
  6,101 confirmed → 6,040 after the temporal cutoff (2026-06-30).
- **Final dataset:** 1,566 rows (date × slot), 2024-05-09 to 2026-06-30.
- **Chronological train/test split** (done before any EDA, to avoid temporal
  leakage): train up to 2026-01-24 (1,252 rows), test from 2026-01-25 to
  2026-06-30 (314 rows).

### Solution

**Supervised regression on a time series.** Target: `n_citas` per date × time slot.

Pipeline: raw-to-grid transformation → EDA → feature engineering (trend, day
grouping, season, official holidays, commercial dates — all validated against the
target) → preprocessing (closed-day exclusion, one-hot encoding, scaling) →
modeling (MAE as primary metric, 6-model comparison under `TimeSeriesSplit`
cross-validation, hyperparameter search with `RandomizedSearchCV`) → single final
evaluation against the untouched test set. Fully assembled in
[`main.ipynb`](main.ipynb).

### Repository structure

See the Spanish section above — same structure, following the course's required
`src/`-based layout.

### Technologies

Python (pandas, numpy), scikit-learn, matplotlib, seaborn, holidays, joblib,
Jupyter, Git/GitHub (Git Flow with reviewed Pull Requests and issue tracking).

### Reproduction

```bash
git clone https://github.com/Emigarsan/ML_Spa_M003.git
cd ML_Spa_M003
pip install -r requirements.txt
```

Running `main.ipynb` end-to-end requires your own local copy of the confidential
`informe_NOUP.csv` export (see Dataset above) — without it, the first stage fails.
To use the already-trained model directly, load
`src/models/modelo_ocupacion.joblib` and `src/models/scaler.joblib` with `joblib`
(see the Spanish section for the exact snippet).

### Main results

Final model: **RandomForest** (optimized via `RandomizedSearchCV` under temporal
cross-validation). Single evaluation against test (2026-01-25 → 2026-06-30):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Mean baseline | 2.76 | 3.70 | −0.22 |
| Seasonal baseline (t-7) | 2.46 | 3.16 | 0.11 |
| **RandomForest (final)** | **1.92** | **2.52** | **0.43** |

Test MAE (1.92) is consistent with the cross-validated estimate (1.74) — the
temporal validation methodology held up, without severe overfitting. Top features:
time trend, afternoon slot, day of week. An MAE of ~2 appointments against a mean
demand of ~5 is enough to size staffing by band (low/normal/peak), but not for
per-room assignment. See the Spanish section for the full limitations and
improvement roadmap.

### Authors

- **Emilio Garrote** — [github.com/Emigarsan](https://github.com/Emigarsan)
- **lolarealcejudo27** — [github.com/lolarealcejudo27](https://github.com/lolarealcejudo27)
- **MCCFern** — [github.com/MCCFern](https://github.com/MCCFern)

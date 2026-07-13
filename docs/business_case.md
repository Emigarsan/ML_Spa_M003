# Business Case — Predicción de ocupación del spa

Fase 1 de la guía orientativa (pasos 1-7). Documenta el problema de negocio y la
justificación del enfoque de ML *antes* de entrar en modelado, y sirve de base
para las primeras celdas del `main.ipynb` final.

## 1. Business understanding

El spa (Sevilla) gestiona su día a día por **tramos horarios** (mañana/tarde,
corte a las 14:00). La decisión de negocio que un modelo puede mejorar es
**cuánto personal y cuántas cabinas dejar preparadas en cada tramo**: hoy esa
decisión se toma por intuición o por la media histórica, sin anticipar picos
ni valles.

Un modelo que prediga la demanda del tramo siguiente permite:

- Evitar sobre-dotación de personal en tramos flojos (coste).
- Evitar infra-dotación en tramos punta (pérdida de reservas o mala experiencia).
- Anticipar la demanda de fechas atípicas (festivos, San Valentín, temporada alta).

## 2. Hipótesis y objetivo de modelado

- **Tipo de problema:** regresión sobre serie temporal.
- **Target:** `n_citas` — número de citas ejecutadas en un tramo horario
  (mañana/tarde) de un día concreto. Es un conteo entero ≥ 0.
- **Hipótesis de partida (confirmadas en el EDA):**
  - La demanda tiene estructura semanal marcada, no gradual (plano
    lunes-jueves, sube el viernes tarde, pico el fin de semana).
  - El tramo de tarde casi dobla al de mañana.
  - Hay estacionalidad anual (invierno arriba, verano abajo) y una tendencia de
    crecimiento real del negocio (mayo se multiplicó ×5,6 entre 2024 y 2025).
  - Existen cierres reales del negocio (Navidad/Reyes y dos bloques en 2025)
    que hay que distinguir de "demanda cero por baja afluencia".

## 3. Plan de acción

El modelo alimenta una **decisión operativa recurrente**, no un informe puntual:
con la predicción de `n_citas` por tramo para los próximos días, el negocio
ajusta el cuadrante de personal y la disponibilidad de cabinas. No sustituye el
criterio humano — lo informa con una cifra objetiva y verificable contra lo
ocurrido.

**Fuera de alcance de este proyecto** (posible extensión futura, no
comprometida): segmentación de clientes por recencia/frecuencia/gasto (RFM).
Se valoró en el estudio de viabilidad inicial como complemento no supervisado,
pero el equipo no ha empezado trabajo sobre ello — el foco es el modelo de
ocupación.

## 4. Requerimientos de los datos

- Serie histórica de citas ejecutadas con fecha y franja horaria.
- Variables de calendario derivables de la fecha (día de la semana, mes,
  festivos) — no requieren dato adicional del negocio.
- Mínimo deseable: al menos un ciclo anual completo para poder validar
  estacionalidad (se dispone de ~2 años).

## 5. Disponibilidad

Datos disponibles y accesibles desde el propio sistema de reservas del spa
(exportes en CSV). Cubren mayo 2024 – junio 2026 a nivel de ticket individual,
de los que se deriva la serie fecha × tramo usada para modelar.

## 6. Adquisición de datos

Exportes reales del sistema de reservas (no un dataset público): informes de
ventas y reservas con una fila por pago/reserva, transformados en
`transform.ipynb` a una rejilla fecha × tramo con el conteo de citas y
variables de calendario. El fichero crudo con datos personales (nombre,
teléfono, email) se procesa en local y **no se versiona** — solo se sube el
agregado sin PII (`data/processed/`).

## 7. Calidad

Validado en el estudio de viabilidad y confirmado en el EDA:

- Serie casi continua: 802 días distintos con actividad de un total de 788
  días naturales en el rango, sin huecos relevantes.
- Identidad y consistencia de las reservas verificadas (deduplicación de pagos
  múltiples por reserva, filtrado de canceladas).
- Dataset limpio: pocos outliers y explicables como demanda real en fechas
  pico, no como errores de captura.
- **Aviso de métrica:** el MAPE no es viable como métrica principal — un 21%
  de los tramos de mañana tienen `n_citas = 0`, lo que dispara el error
  porcentual a infinito. Se usa MAE (principal) y RMSE (secundaria); ver
  `notebooks/modeling.ipynb`.
- **Aviso de censura:** los datos están cortados a la fecha de exportación
  (30-jun-2026); con una antelación de reserva mediana de 1 día, solo las
  últimas ~2 semanas de la serie están infra-contadas.

---
*Referencias: `notebooks/transform.ipynb` y `eda.ipynb` (rama `develop`),
estudio de viabilidad previo a la planning session.*

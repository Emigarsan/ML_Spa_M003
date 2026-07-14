# src/transform.py
"""
Funciones para transformar el export bruto de reservas (informe_NOUP.csv) en
el dataset limpio fecha x tramo con el target n_citas. Se usan para
reconstruir el histórico cuando llega un export nuevo — no intervienen en la
predicción de una fecha futura suelta, eso lo cubre `src/feature_engineering.py`.

Cada función corresponde a un paso de `transform.ipynb`, para poder seguir
viendo el resultado intermedio de cada uno (y su print de verificación) sin
tener que ejecutar todo el pipeline de una vez.
"""
import re

import numpy as np
import pandas as pd

from src.utils.feature_engineering import anadir_variables_calendario, tramo_desde_hora

NON_SERVICE_PRODUCTS = ['¡Tarjeta de regalo!', 'Membresías']

DISP_RE = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4})(?:\s+a las\s+(\d{1,2}:\d{2})(?:\s*[-–—]\s*(\d{1,2}:\d{2}))?)?'
)


def parse_disponibilidad(valor):
    """Extrae (fecha, hora_inicio, hora_fin) del texto de la columna 'Disponibilidad'."""
    if pd.isna(valor):
        return pd.NaT, None, None
    m = DISP_RE.match(str(valor))
    if not m:
        return pd.NaT, None, None
    fecha_str, hora_inicio, hora_fin = m.groups()
    try:
        fecha = pd.to_datetime(fecha_str, format='%d/%m/%y')
    except ValueError:
        fecha = pd.NaT
    return fecha, hora_inicio, hora_fin


def cargar_csv_bruto(raw_path):
    """Paso 1: carga informe_NOUP.csv y descarta la fila de totales del propio export."""
    df_raw = pd.read_csv(raw_path, skiprows=1, encoding='utf-8-sig')
    df_raw = df_raw[df_raw['ID de reserva'].notna()].copy()
    return df_raw


def parsear_fechas(df_raw):
    """Paso 2: añade fecha_cita, hora_inicio y hora_fin a partir de 'Disponibilidad'."""
    df_raw = df_raw.copy()
    parsed = df_raw['Disponibilidad'].apply(parse_disponibilidad)
    df_raw['fecha_cita'] = parsed.apply(lambda x: x[0])
    df_raw['hora_inicio'] = parsed.apply(lambda x: x[1])
    df_raw['hora_fin'] = parsed.apply(lambda x: x[2])
    return df_raw


def filtrar_reservas_servicio(df_raw):
    """Paso 3: excluye tarjetas de regalo/membresías y deduplica a nivel de reserva."""
    servicios = df_raw[~df_raw['Producto'].isin(NON_SERVICE_PRODUCTS)].copy()
    reservas = (
        servicios.sort_values('Creado el')
                 .drop_duplicates(subset='ID de reserva', keep='first')
                 .copy()
    )
    return reservas


def filtrar_confirmadas(reservas):
    """Paso 4: solo reservas no canceladas y con fecha/hora de cita válida."""
    confirmadas = reservas[reservas['¿Cancelado?'] == 'No'].copy()
    confirmadas = confirmadas.dropna(subset=['fecha_cita', 'hora_inicio'])
    return confirmadas


def aplicar_corte_temporal(confirmadas, fecha_corte):
    """Paso 5: descarta citas posteriores a fecha_corte (demanda todavía en curso o futura)."""
    return confirmadas[confirmadas['fecha_cita'] <= fecha_corte].copy()


def asignar_tramo(confirmadas):
    """Paso 6: tramo horario (mañana/tarde) a partir de hora_inicio."""
    confirmadas = confirmadas.copy()
    confirmadas['tramo'] = confirmadas['hora_inicio'].apply(tramo_desde_hora)
    return confirmadas


def construir_rejilla(confirmadas, fecha_corte):
    """
    Pasos 7-8: rejilla completa fecha x tramo con el target `n_citas` (0
    donde no hubo ninguna reserva), más las variables de calendario básicas.
    Las variables derivadas de la EDA (grupo_dia, temporada, festivos...) se
    añaden después, en feature_engineering.ipynb — ver la nota de
    `construir_features_df` en `src/feature_engineering.py` sobre por qué
    están separadas.
    """
    conteo = confirmadas.groupby(['fecha_cita', 'tramo']).size().rename('n_citas').reset_index()

    fechas = pd.date_range(confirmadas['fecha_cita'].min(), fecha_corte, freq='D')
    rejilla = pd.MultiIndex.from_product(
        [fechas, ['mañana', 'tarde']], names=['fecha_cita', 'tramo']
    ).to_frame(index=False)

    dataset = rejilla.merge(conteo, on=['fecha_cita', 'tramo'], how='left')
    dataset['n_citas'] = dataset['n_citas'].fillna(0).astype(int)

    dataset = anadir_variables_calendario(dataset)
    return dataset.sort_values(['fecha_cita', 'tramo']).reset_index(drop=True)


def split_train_test(dataset, test_size=0.2):
    """Paso 11: split cronológico (nunca aleatorio) train/test."""
    fechas_unicas = dataset['fecha_cita'].drop_duplicates().sort_values().reset_index(drop=True)
    n_test_dias = int(np.ceil(len(fechas_unicas) * test_size))
    fecha_split = fechas_unicas.iloc[-n_test_dias]

    train = dataset[dataset['fecha_cita'] < fecha_split].copy()
    test = dataset[dataset['fecha_cita'] >= fecha_split].copy()
    return train, test, fecha_split

# src/preprocessing.py
import pandas as pd
import numpy as np

def mapear_grupo_dia(nombre_dia):
    """Agrupa los días según el comportamiento de la demanda detectado en el EDA."""
    if nombre_dia in ['Monday', 'Tuesday', 'Wednesday', 'Thursday']:
        return 'Lunes-Jueves'
    elif nombre_dia == 'Friday':
        return 'Viernes'
    elif nombre_dia in ['Saturday', 'Sunday']:
        return 'FinDeSemana'
    return 'Desconocido'

def mapear_temporada(mes):
    """Agrupa los meses por estaciones o comportamiento del negocio."""
    if mes in [12, 1, 2]:
        return 'Invierno_Pico'
    elif mes in [6, 7, 8]:
        return 'Verano_Valle'
    else:
        return 'Media_Temporada'

def build_features(df):
    """
    Transforma el dataset bruto aplicando la ingeniería de características.
    Elimina columnas que no aportan información lineal al modelo de forma segura.
    """
    data = df.copy()
    
    # Asegurar formato fecha
    data['fecha_cita'] = pd.to_datetime(data['fecha_cita'])
    
    # Variable de tendencia: Días transcurridos desde el inicio real del negocio (Mayo 2024)
    fecha_min_global = pd.to_datetime("2024-05-09")
    data['dias_desde_inicio'] = (data['fecha_cita'] - fecha_min_global).dt.days
    
    # Aplicar agrupaciones estratégicas del EDA si existen las columnas base
    if 'nombre_dia' in data.columns:
        data['grupo_dia'] = data['nombre_dia'].apply(mapear_grupo_dia)
    if 'mes' in data.columns:
        data['temporada'] = data['mes'].apply(mapear_temporada)
    
    # Convertir variable tramo a binaria (mañana = 0, tarde = 1)
    if 'tramo' in data.columns:
        data['tramo_tarde'] = data['tramo'].map({'mañana': 0, 'tarde': 1})
    
    # One-Hot Encoding manual/sencillo para las nuevas categorías creadas
    columnas_encoding = [col for col in ['grupo_dia', 'temporada'] if col in data.columns]
    if columnas_encoding:
        data = pd.get_dummies(data, columns=columnas_encoding, drop_first=True)
    
    # Separar Target (y) de Características (X) de forma segura
    if 'n_citas' in data.columns:
        y = data['n_citas']
    else:
        y = None
        
    # Identificar qué columnas de control/originales existen realmente en este paso para borrarlas
    columnas_posibles = ['fecha_cita', 'tramo', 'nombre_dia', 'dia_semana', 'mes', 'anio', 'semana_iso', 'n_citas']
    columnas_a_eliminar = [col for col in columnas_posibles if col in data.columns]
    
    X = data.drop(columns=columnas_a_eliminar)
    
    # Convertir booleanos resultantes (del encoding) a enteros (0 y 1)
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
        
    return X, y
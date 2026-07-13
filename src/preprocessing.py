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
    Transforma el dataset bruto aplicando ingeniería de características.
    Filtra días de cierre y elimina columnas redundantes.
    """
    data = df.copy()
    
    #Excluir días de cierre del entrenamiento
    if 'es_cierre' in data.columns:
        data = data[data['es_cierre'] != 1].reset_index(drop=True)
    
    #Asegurar formato fecha
    data['fecha_cita'] = pd.to_datetime(data['fecha_cita'])
    
    #Variable de tendencia: Días transcurridos desde el inicio (Mayo 2024)
    fecha_min_global = pd.to_datetime("2024-05-09")
    data['dias_desde_inicio'] = (data['fecha_cita'] - fecha_min_global).dt.days
    
    #One-Hot Encoding para las nuevas categorías estratégicas
    columnas_encoding = [col for col in ['grupo_dia', 'temporada'] if col in data.columns]
    if columnas_encoding:
        data = pd.get_dummies(data, columns=columnas_encoding, drop_first=True)
    
    #Separar Target (y) de Características (X) de forma segura
    if 'n_citas' in data.columns:
        y = data['n_citas']
    else:
        y = None
        
  #Eliminación de columnas redundantes o repetidas
    columnas_a_eliminar = [
        'fecha_cita', 'tramo', 'nombre_dia', 'es_finde', 'es_cierre',
        'dia_semana', 'mes', 'anio', 'semana_iso', 'n_citas'
    ]
    
    #Filtramos para borrar solo las que realmente existan en el DataFrame actual
    columnas_reales_a_eliminar = [col for col in columnas_a_eliminar if col in data.columns]
    X = data.drop(columns=columnas_reales_a_eliminar)
    
    #Convertir booleanos resultantes (del encoding) a enteros (0 y 1)
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
        
    return X, y  
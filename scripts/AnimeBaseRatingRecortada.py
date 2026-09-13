import pandas as pd
import numpy as np

datos = pd.read_csv('Bases/anime.csv')

# 1. Seleccionar una muestra aleatoria de 500 usuarios únicos
# ALEATORIO, agregar random_state=semilla  para fijar semilla 
usuarios_muestra = datos['members'].drop_duplicates().sample(n=500)

# 2. Filtrar el DataFrame original para quedarnos solo con esos 500 usuarios
datos_recortados = datos[datos['members'].isin(usuarios_muestra)]

# 3. Crear la matriz de ratings con los datos recortados
matriz_ratings = datos_recortados.pivot_table(
    index='members', 
    columns='anime_id', 
    values='rating'
)

# Proporción de datos no observados
total_nans = matriz_ratings.isna().sum().sum()
total_celdas = matriz_ratings.size
proporcion_nans = total_nans / total_celdas
print(f"Proporción de valores no observados: {proporcion_nans:.4f}")
print(f"Es decir, aproximadamente un {proporcion_nans * 100:.2f}% de la matriz está vacío.")

# Guardar la matriz recortada en formato parquet
matriz_ratings.to_parquet('Bases/anime_ratings_500.parquet')
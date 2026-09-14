"""
Matrix Factorization for Recommender Systems
=============================================

Variante implementada aqui:
    1) Matrix-based Batch Gradient Descent

Todas las funciones trabajan con cualquier matriz de ratings R (numpy array
o pandas DataFrame) donde las entradas faltantes son NaN. Las filas son
usuarios, las columnas son items (peliculas).

CORRECCION respecto a la version original:
    Antes, `user_names` y `movie_names` se generaban con `range(...)`,
    es decir, usaban la POSICION de cada fila/columna en la matriz
    pivoteada (0, 1, 2, ...) en vez del userId / movieId real. Esto hacia
    que las recomendaciones mostraran indices que no correspondian al
    movieId verdadero, imposibilitando cruzarlas con movies.csv.

    Ahora se usan directamente `_R_values.index` y `_R_values.columns`,
    que son los userId y movieId reales que vienen del pivot_table.
"""

import numpy as np
import pandas as pd

datos = pd.read_csv('Bases/ratings.csv')
peliculas = pd.read_csv('Bases/movies.csv')


########################################################################################
#    ACHICAR LA MATRIZ ALEATORIAMENTE


# 1. Seleccionar una muestra aleatoria de 500 usuarios unicos
# ALEATORIO, agregar random_state=semilla para fijar semilla
usuarios_muestra = datos['userId'].drop_duplicates().sample(n=500, random_state=42)

# 2. Filtrar el DataFrame original para quedarnos solo con esos 500 usuarios
datos_recortados = datos[datos['userId'].isin(usuarios_muestra)]

# 3. Crear la matriz de ratings con los datos recortados
matriz_ratings = datos_recortados.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

#######################################################################################


_R_values = matriz_ratings

# numpy's default reshape is row-major, equivalent to R's byrow=TRUE
R = np.array(_R_values, dtype=float)

# --- FIX: usar las etiquetas reales (userId / movieId), no posiciones ---
user_names = list(_R_values.index)
movie_names = list(_R_values.columns)


# ============================================================== #
# Helper: pull raw numpy array + optional row/col labels out of   #
# either a numpy array or a pandas DataFrame, so every algorithm  #
# below works with either input type.                             #
# ============================================================== #

def _as_matrix(R):
    """Return (values, row_labels, col_labels) for R (ndarray or DataFrame)."""
    if isinstance(R, pd.DataFrame):
        return R.values.astype(float), list(R.index), list(R.columns)
    R = np.asarray(R, dtype=float)
    return R, list(range(R.shape[0])), list(range(R.shape[1]))


# ============================================================== #
# 1) Matrix-based Batch Gradient Descent (Aggarwal 2016 style)    #
# ============================================================== #

def matrix_batch_gd(R, k=2, gamma=0.001, max_iter=1000, tol=1e-4, seed=1, verbose=True):
    """
    Matrix factorization via batch gradient descent.

    Update rule (Aggarwal, 2016):
        P <- P + gamma * E_zero %*% Q
        Q <- Q + gamma * t(E_zero) %*% P
    where E_zero is the error matrix (R - P Q^T) with missing entries
    zeroed out, so they don't contribute to the gradient.

    Parameters
    ----------
    R : np.ndarray or pd.DataFrame
        Ratings matrix (users x items), NaN for missing entries.
    k : int
        Number of latent factors.
    gamma : float
        Learning rate.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the change in RMSE between iterations.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress every 100 iterations and on convergence.

    Returns
    -------
    dict with keys: P, Q, rmse_history, iterations
    """
    values, _, _ = _as_matrix(R)
    n_users, n_items = values.shape

    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.1, size=(n_users, k))
    Q = rng.normal(0, 0.1, size=(n_items, k))

    observed = np.argwhere(~np.isnan(values))
    obs_i, obs_j = observed[:, 0], observed[:, 1]
    r_observed = values[obs_i, obs_j]

    rmse_history = []
    prev_rmse = np.inf
    iterations = 0

    for it in range(1, max_iter + 1):
        iterations = it
        R_hat = P @ Q.T
        E = values - R_hat
        E_zero = np.where(np.isnan(E), 0.0, E)

        P_new = P + gamma * (E_zero @ Q)
        Q_new = Q + gamma * (E_zero.T @ P)
        P, Q = P_new, Q_new

        predictions = np.sum(P[obs_i, :] * Q[obs_j, :], axis=1)
        current_rmse = np.sqrt(np.mean((r_observed - predictions) ** 2))
        rmse_history.append(current_rmse)

        if it > 1:
            rmse_change = abs(prev_rmse - current_rmse)
            if rmse_change < tol:
                if verbose:
                    print(f"Converged at iteration {it} - RMSE change: {round(rmse_change, 6)}")
                break
        prev_rmse = current_rmse

        if verbose and it % 100 == 0:
            print(f"Iteration {it} - RMSE: {round(current_rmse, 4)}")

    return {"P": P, "Q": Q, "rmse_history": np.array(rmse_history), "iterations": iterations}


# ============================================================== #
#  Recommendation helper                                          #
# ============================================================== #

def recommend_all_unseen(user_name, R_hat, R_actual, value_name="Rating"):
    """
    List all items a user has NOT rated, sorted by predicted score
    (descending).

    Parameters
    ----------
    user_name : row label identifying the user (userId real).
    R_hat : pd.DataFrame or np.ndarray
        Predicted ratings matrix (users x items).
    R_actual : pd.DataFrame or np.ndarray
        Original ratings matrix, same shape/labels as R_hat, with NaN
        for unseen items.
    value_name : str
        Column name to use for the predicted score in the output.

    Returns
    -------
    pd.DataFrame with columns [movieId, value_name]
    """
    if not isinstance(R_hat, pd.DataFrame):
        R_hat = pd.DataFrame(R_hat)
    if not isinstance(R_actual, pd.DataFrame):
        R_actual = pd.DataFrame(R_actual)

    pred = R_hat.loc[user_name]
    seen = R_actual.loc[user_name].notna()
    unseen = pred[~seen]
    unseen_sorted = unseen.sort_values(ascending=False)

    return pd.DataFrame({
        "movieId": unseen_sorted.index,
        value_name: unseen_sorted.round(2).values,
    })


# ============================================================== #
#  Demo / example usage                                           #
# ============================================================== #

if __name__ == "__main__":
    R_df = pd.DataFrame(R, index=user_names, columns=movie_names)

    print("=" * 60)
    print("1) Matrix-based Batch Gradient Descent")
    print("=" * 60)
    result = matrix_batch_gd(R_df)
    R_hat = pd.DataFrame(result["P"] @ result["Q"].T, index=user_names, columns=movie_names)

    # Elegimos el primer usuario real de la muestra (userId real, no posicion 0)
    primer_usuario = user_names[1]

    recs = recommend_all_unseen(primer_usuario, R_hat, R_df).head(10)
    # Cruce con los titulos reales de movies.csv
    recs = recs.merge(peliculas[['movieId', 'title', 'genres']], on='movieId', how='left')

    print()
    print(f"Top 10 recomendaciones para userId = {primer_usuario}:")
    print(recs.to_string(index=False))

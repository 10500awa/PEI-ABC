import numpy as np
import pandas as pd


import numpy as np
import pandas as pd

datos = pd.read_csv('Bases/ratings.csv')
peliculas = pd.read_csv('Bases/movies.csv')

# 1. Seleccionar una muestra aleatoria de 500 usuarios unicos
# ALEATORIO, agregar random_state=semilla para fijar semilla
usuarios_muestra = datos['userId']
#.drop_duplicates().sample(n=500, random_state=42)

# 2. Filtrar el DataFrame original para quedarnos solo con esos 500 usuarios
datos_recortados = datos[datos['userId'].isin(usuarios_muestra)]

# 3. Crear la matriz de ratings con los datos recortados
matriz_ratings = datos_recortados.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

_R_values = matriz_ratings

# numpy's default reshape is row-major, equivalent to R's byrow=TRUE
R = np.array(_R_values, dtype=float)

# --- FIX: usar las etiquetas reales (userId / movieId), no posiciones ---
user_names = list(_R_values.index)
movie_names = list(_R_values.columns)


#######################################################################################


def _as_matrix(R):
    """Return (values, row_labels, col_labels) for R (ndarray or DataFrame)."""
    if isinstance(R, pd.DataFrame):
        return R.values.astype(float), list(R.index), list(R.columns)
    R = np.asarray(R, dtype=float)
    return R, list(range(R.shape[0])), list(range(R.shape[1]))

# ============================================================== #
#   4) SGD Algorithm with regularization                          #
# ============================================================== #

def sgd_aggarwal_reg(R, k=2, gamma=0.01, lambda_=0.1, max_iter=1000, tol=1e-4, seed=1, verbose=True):
    """
    Stochastic gradient descent matrix factorization with L2 regularization.

    For each observed entry (i, j):
        e_ij <- r_ij - p_i . q_j
        p_i  <- p_i + gamma * (e_ij * q_j - lambda * p_i)
        q_j  <- q_j + gamma * (e_ij * p_i - lambda * q_j)    (uses UPDATED p_i)

    Returns
    -------
    dict with keys: P, Q, R_hat
    """
    values, _, _ = _as_matrix(R)
    n_users, n_items = values.shape

    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.1, size=(n_users, k))
    Q = rng.normal(0, 0.1, size=(n_items, k))

    observed = np.argwhere(~np.isnan(values))
    n_obs = observed.shape[0]

    prev_rmse = np.inf

    for it in range(1, max_iter + 1):
        shuffle = rng.permutation(n_obs)

        for idx in shuffle:
            i, j = observed[idx]
            r_ij = values[i, j]
            r_hat = np.dot(P[i, :], Q[j, :])
            e = r_ij - r_hat

            P[i, :] = P[i, :] + gamma * (e * Q[j, :] - lambda_ * P[i, :])
            Q[j, :] = Q[j, :] + gamma * (e * P[i, :] - lambda_ * Q[j, :])

        obs_i, obs_j = observed[:, 0], observed[:, 1]
        predictions = np.sum(P[obs_i, :] * Q[obs_j, :], axis=1)
        current_rmse = np.sqrt(np.mean((values[obs_i, obs_j] - predictions) ** 2))

        if abs(prev_rmse - current_rmse) < tol:
            if verbose:
                print(f"Converged at iteration {it} - RMSE change: {round(abs(prev_rmse - current_rmse), 6)}")
            break

        prev_rmse = current_rmse

        if verbose and it % 10 == 0:
            print(f"Iteration {it} - RMSE: {round(current_rmse, 4)}")

    return {"P": P, "Q": Q, "R_hat": P @ Q.T}


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
#  Demo / example usage (mirrors the original R script)           #
# ============================================================== #

if __name__ == "__main__":
    R_df = pd.DataFrame(R, index=user_names, columns=movie_names)


    print()
    print("=" * 60)
    print("4) SGD Algorithm with regularization")
    print("=" * 60)
    result_sgd_reg = sgd_aggarwal_reg(R_df)
    R_hat_sgd_reg = pd.DataFrame(result_sgd_reg["R_hat"], index=user_names, columns=movie_names)
    print()
    # Elegimos el primer usuario real de la muestra
    primer_usuario = user_names[0]

    recs = recommend_all_unseen(primer_usuario, R_hat_sgd_reg, R_df).head(10)
    # Cruce con los titulos reales de movies.csv
    recs = recs.merge(peliculas[['movieId', 'title', 'genres']], on='movieId', how='left')

    print()
    print(f"Top 10 recomendaciones para userId = {primer_usuario}:")
    print(recs.to_string(index=False))


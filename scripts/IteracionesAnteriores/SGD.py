"""
Matrix Factorization for Recommender Systems
=============================================

Python translation of an R script implementing four variants of
matrix-factorization-based collaborative filtering (Aggarwal, 2016 style):

    1) Matrix-based Batch Gradient Descent
    2) Stochastic Gradient Descent (SGD)
    3) Matrix-based Batch Gradient Descent with L2 regularization
    4) Stochastic Gradient Descent (SGD) with L2 regularization

All functions work with any ratings matrix R (numpy array or pandas
DataFrame) where missing entries are represented as NaN. Rows are
users, columns are items.
"""

import numpy as np
import pandas as pd


# ============================================================== #
#  Toy example matrix (movies), same data as the original R code #
# ============================================================== #

_R_values = [
    4, np.nan, np.nan, 2, np.nan, 5,
    np.nan, 1, 5, np.nan, 2, np.nan,
    2, 5, 3, 4, np.nan, np.nan,
    np.nan, np.nan, np.nan, 5, 4, 3,
    np.nan, 2, np.nan, np.nan, 3, 5,
    1, np.nan, 2, 4, 5, np.nan,
]

# numpy's default reshape is row-major, equivalent to R's byrow=TRUE
R = np.array(_R_values, dtype=float).reshape(6, 6)

user_names = [f"Usuario {i}" for i in range(1, 7)]
movie_names = [
    "Tonto y re tonto",
    "El padrino",
    "La pistola desnuda",
    "Buenos Muchachos",
    "Casino",
    "Zoolander",
]


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
#   2) SGD Algorithm based on Aggarwal (2016)                     #
# ============================================================== #

def sgd_aggarwal(R, k=2, gamma=0.01, max_iter=1000, tol=1e-4, seed=42, verbose=True):
    """
    Matrix factorization via stochastic gradient descent.

    For each observed entry (i, j), visited in random order each epoch:
        e_ij   <- r_ij - p_i . q_j
        p_i    <- p_i + gamma * e_ij * q_j
        q_j    <- q_j + gamma * e_ij * p_i     (uses the UPDATED p_i,
                                                 matching the original R code)

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

            P[i, :] = P[i, :] + gamma * e * Q[j, :]
            Q[j, :] = Q[j, :] + gamma * e * P[i, :]

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
    user_name : row label (or integer position if R_hat/R_actual are
        plain numpy arrays without an index) identifying the user.
    R_hat : pd.DataFrame or np.ndarray
        Predicted ratings matrix (users x items).
    R_actual : pd.DataFrame or np.ndarray
        Original ratings matrix, same shape/labels as R_hat, with NaN
        for unseen items.
    value_name : str
        Column name to use for the predicted score in the output.

    Returns
    -------
    pd.DataFrame with columns [item label column, value_name]
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
        "Item": unseen_sorted.index,
        value_name: unseen_sorted.round(2).values,
    })


# ============================================================== #
#  Demo / example usage (mirrors the original R script)           #
# ============================================================== #

if __name__ == "__main__":
    R_df = pd.DataFrame(R, index=user_names, columns=movie_names)

    print("=" * 60)
    print("1) Matrix-based Batch Gradient Descent")
    print("=" * 60)
    result = matrix_batch_gd(R_df)
    R_hat = pd.DataFrame(result["P"] @ result["Q"].T, index=user_names, columns=movie_names)
    print(R_hat.round(2))
    print()
    print(recommend_all_unseen("Usuario 1", R_hat, R_df))

    print()
    print("=" * 60)
    print("2) SGD Algorithm (Aggarwal 2016)")
    print("=" * 60)
    result_sgd = sgd_aggarwal(R_df)
    R_hat_sgd = pd.DataFrame(result_sgd["R_hat"], index=user_names, columns=movie_names)
    print(R_hat_sgd.round(2))
    print()
    print(recommend_all_unseen("Usuario 2", R_hat_sgd, R_df))

    print()
    print("=" * 60)
    print("3) Matrix-based Batch Gradient Descent with regularization")
    print("=" * 60)
    result_reg = matrix_batch_gd_reg(R_df, k=2, gamma=0.01, lambda_=0.1, max_iter=1000, tol=1e-4)
    R_hat_gd_reg = pd.DataFrame(result_reg["P"] @ result_reg["Q"].T, index=user_names, columns=movie_names)
    print(recommend_all_unseen("Usuario 6", R_hat_gd_reg, R_df))

    print()
    print("=" * 60)
    print("4) SGD Algorithm with regularization")
    print("=" * 60)
    result_sgd_reg = sgd_aggarwal_reg(R_df)
    R_hat_sgd_reg = pd.DataFrame(result_sgd_reg["R_hat"], index=user_names, columns=movie_names)
    print(recommend_all_unseen("Usuario 6", R_hat_sgd_reg, R_df))

    print()
    print("=" * 60)
    print("SGD with regularization, k=16 latent factors")
    print("=" * 60)
    result_k16 = sgd_aggarwal_reg(R_df, k=16, gamma=0.01, lambda_=0.1, max_iter=1000, tol=1e-4)
    R_hat_k16 = pd.DataFrame(result_k16["R_hat"], index=user_names, columns=movie_names)
    print(recommend_all_unseen("Usuario 1", R_hat_k16, R_df))

"""
Matrix factorization via stochastic gradient descent (SGD) with L2
regularization, plus k-fold cross-validation over a range of latent
factor counts.
"""

import numpy as np
import pandas as pd
datos = pd.read_csv('Bases/ratings.csv')
peliculas = pd.read_csv('Bases/movies.csv')


########################################################################################
#    ACHICAR LA MATRIZ ALEATORIAMENTE


# 1. Seleccionar una muestra aleatoria de 500 usuarios unicos
# ALEATORIO, agregar random_state=semilla para fijar semilla
usuarios_muestra = datos['userId']
#.drop_duplicates().sample(n=500, random_state=42)

# 2. Filtrar el DataFrame original para quedarnos solo con esos 500 usuarios
datos_recortados = datos[datos['userId'].isin(usuarios_muestra)]

# 3. Crear la matriz de ratings con los datos recortados
df_rating = datos_recortados.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

#######################################################################################

# Convert to a plain numpy matrix (rownames/userId index is dropped here,
# just like column_to_rownames("userId") |> as.matrix() in R)
R = df_rating.to_numpy()


# ============================================================================
#              4) SGD Algorithm with regularization
# ============================================================================

def sgd_aggarwal_reg(R, k=2, gamma=0.01, lambda_=0.1, max_iter=1000, tol=1e-4,
                      seed=1, verbose=True):
    """
    Matrix factorization via SGD with L2 regularization (Aggarwal's
    update rule).

    Parameters
    ----------
    R : np.ndarray
        Ratings matrix with np.nan for missing entries.
    k : int
        Number of latent factors.
    gamma : float
        Learning rate.
    lambda_ : float
        Regularization strength (named lambda_ since `lambda` is a
        reserved keyword in Python).
    max_iter : int
        Maximum number of epochs (full passes over observed entries).
    tol : float
        Convergence tolerance on the change in RMSE between epochs.
    seed : int
        Random seed, for reproducibility (analogous to set.seed(42) in R).
    verbose : bool
        Whether to print progress, mirroring the cat() calls in R.

    Returns
    -------
    dict with keys 'P', 'Q', 'R_hat'
    """
    rng = np.random.default_rng(seed)

    n_users, n_items = R.shape

    # Initialize latent factor matrices
    P = rng.normal(0, 0.1, size=(n_users, k))
    Q = rng.normal(0, 0.1, size=(n_items, k))

    # Get observed (non-missing) entries
    observed_rows, observed_cols = np.where(~np.isnan(R))
    n_obs = len(observed_rows)

    prev_rmse = np.inf

    for iteration in range(1, max_iter + 1):
        # Shuffle the order in which observed entries are visited this epoch
        shuffle = rng.permutation(n_obs)

        for idx in shuffle:
            i = observed_rows[idx]
            j = observed_cols[idx]

            r_ij = R[i, j]
            r_hat = P[i, :] @ Q[j, :]
            e = r_ij - r_hat

            # Sequential update: note that the Q update below uses the
            # *already updated* P[i, :], exactly matching the R code
            # (which reassigns P[i, ] on the line before updating Q[j, ]).
            P[i, :] = P[i, :] + gamma * (e * Q[j, :] - lambda_ * P[i, :])
            Q[j, :] = Q[j, :] + gamma * (e * P[i, :] - lambda_ * Q[j, :])

        # Check convergence after each full pass (epoch): compute RMSE
        # on all observed entries
        predictions = np.sum(P[observed_rows, :] * Q[observed_cols, :], axis=1)
        current_rmse = np.sqrt(np.mean((R[observed_rows, observed_cols] - predictions) ** 2))

        if abs(prev_rmse - current_rmse) < tol:
            if verbose:
                print(f"Converged at iteration {iteration} - RMSE change: "
                      f"{abs(prev_rmse - current_rmse):.6f}")
            break

        prev_rmse = current_rmse

        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration} - RMSE: {current_rmse:.4f}")

    return {"P": P, "Q": Q, "R_hat": P @ Q.T}


# ============================================================================
# K-FOLD CROSS-VALIDATION FOR SELECTING NUMBER OF LATENT FACTORS
# ============================================================================

def kfold_cv_mf(R, k_folds=10, factors_range=None, gamma=0.01, lambda_=0.1,
                 max_iter=500, tol=1e-4, seed=1, verbose=True):
    """
    K-fold cross-validation for matrix factorization: evaluates several
    candidate numbers of latent factors and reports mean/SD RMSE across
    folds for each.

    Parameters mirror the R function `kfold_cv_mf`.

    Returns
    -------
    pandas.DataFrame with columns: k_factors, mean_rmse, sd_rmse, fold_rmse
    (fold_rmse holds a list of per-fold RMSE values, like the list-column
    in the R data frame).
    """
    if factors_range is None:
        factors_range = [2, 5, 8, 10]

    # Get observed (non-missing) entries
    observed_rows, observed_cols = np.where(~np.isnan(R))
    n_obs = len(observed_rows)

    # Shuffle observations and assign to folds (random fold assignment,
    # analogous to the shuffle + rep(1:k_folds) + reorder trick in R)
    rng = np.random.default_rng(seed)
    fold_pattern = np.resize(np.arange(1, k_folds + 1), n_obs)
    fold_assignment = rng.permutation(fold_pattern)

    results = []

    if verbose:
        print(f"Starting {k_folds}-fold cross-validation")
        print("Testing factors:", ", ".join(str(f) for f in factors_range))
        print()

    for k in factors_range:
        if verbose:
            print(f"=== Testing k = {k} factors ===")

        fold_rmse = np.zeros(k_folds)

        for fold in range(1, k_folds + 1):
            test_mask = fold_assignment == fold
            test_rows = observed_rows[test_mask]
            test_cols = observed_cols[test_mask]

            # Create train matrix: mask out the test entries with NaN
            R_train = R.copy()
            R_train[test_rows, test_cols] = np.nan

            # Train model on the training data
            model = sgd_aggarwal_reg(
                R_train, k=k, gamma=gamma, lambda_=lambda_,
                max_iter=max_iter, tol=tol, seed=seed, verbose=False,
            )

            # Predict on the held-out test entries
            R_hat = model["R_hat"]
            predictions = R_hat[test_rows, test_cols]
            actual = R[test_rows, test_cols]

            # RMSE on this test fold
            fold_rmse[fold - 1] = np.sqrt(np.mean((actual - predictions) ** 2))

            if verbose:
                print(f"  Fold {fold}/{k_folds} - RMSE: {fold_rmse[fold - 1]:.4f}")

        results.append({
            "k_factors": k,
            "mean_rmse": fold_rmse.mean(),
            "sd_rmse": fold_rmse.std(ddof=1),   # ddof=1 to match R's sd()
            "fold_rmse": fold_rmse.tolist(),
        })

        if verbose:
            print(f"  --> Mean RMSE: {fold_rmse.mean():.4f} "
                  f"(SD: {fold_rmse.std(ddof=1):.4f})\n")

    return pd.DataFrame(results)


# ----------------------------------------------------------------------------
# Fixed sequence of dimensions to test
# ----------------------------------------------------------------------------

factors_range_large = [2, 5, 8, 10, 20, 30] + list(range(50, 501, 50))

# ----------------------------------------------------------------------------
# Run the cross-validation
# ----------------------------------------------------------------------------

cv_results = kfold_cv_mf(
    R,
    k_folds=10,
    factors_range=factors_range_large,
    gamma=0.01,
    lambda_=0.1,
    max_iter=1000,
)

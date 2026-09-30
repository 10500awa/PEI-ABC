"""
Matrix factorization via stochastic gradient descent (SGD) with L2
regularization, plus k-fold cross-validation over a GRID of
(latent factors, regularization strength lambda), with detailed progress
reporting (errors, timing, epochs, convergence, ETA).
"""

import itertools
import time

import numpy as np
import pandas as pd

# numba is optional, but makes the SGD loop ~50-100x faster.
# Install with:  pip install numba
try:
    from numba import njit
except ImportError:
    njit = None


def _fmt_time(seconds):
    """Human-readable duration: 3.2s, 4m 05.1s, 1h 12m."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


# ============================================================================
#              1) One SGD epoch (compiled with numba if available)
# ============================================================================

def _sgd_epoch(P, Q, rows, cols, vals, order, gamma, lambda_):
    """One pass over the observed ratings, in the given order (in place)."""
    for idx in order:
        i = rows[idx]
        j = cols[idx]
        e = vals[idx] - np.dot(P[i, :], Q[j, :])

        # Sequential update: the Q update uses the already-updated P[i, :]
        P[i, :] = P[i, :] + gamma * (e * Q[j, :] - lambda_ * P[i, :])
        Q[j, :] = Q[j, :] + gamma * (e * P[i, :] - lambda_ * Q[j, :])


if njit is not None:
    _sgd_epoch = njit(cache=True)(_sgd_epoch)


def _warmup():
    """Trigger numba compilation on a tiny matrix so it doesn't pollute timings."""
    R_small = np.array([[1.0, np.nan], [np.nan, 2.0]])
    sgd_aggarwal_reg(R_small, k=2, max_iter=1, verbose=False, compute_rhat=False)


# ============================================================================
#              2) SGD algorithm with L2 regularization
# ============================================================================

def sgd_aggarwal_reg(R, k=2, gamma=0.01, lambda_=0.1, max_iter=1000, tol=1e-4,
                     seed=1, verbose=True, compute_rhat=True):
    """
    Matrix factorization via SGD with L2 regularization.

    Parameters
    ----------
    R : np.ndarray
        Ratings matrix with np.nan for missing entries.
    k : int
        Number of latent factors.
    gamma : float
        Learning rate.
    lambda_ : float
        L2 regularization strength.
    max_iter : int
        Maximum number of epochs.
    tol : float
        Convergence tolerance on the change in training RMSE between epochs.
    seed : int
        Random seed.
    verbose : bool
        Print progress (every 10 epochs) and an end-of-training summary.
    compute_rhat : bool
        If True, also return the full P @ Q.T matrix (skip it in CV, where
        only the held-out entries are needed).

    Returns
    -------
    dict with keys:
        'P', 'Q'      : latent factor matrices
        'R_hat'       : P @ Q.T (or None)
        'train_rmse'  : final training RMSE
        'rmse_start'  : training RMSE at random initialization (epoch 0)
        'n_iter'      : epochs actually run
        'converged'   : True if stopped because ΔRMSE < tol
        'last_delta'  : |RMSE change| in the last epoch
        'history'     : list with the training RMSE after each epoch
        'elapsed'     : training time in seconds
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape

    P = rng.normal(0, 0.1, size=(n_users, k))
    Q = rng.normal(0, 0.1, size=(n_items, k))

    rows, cols = np.where(~np.isnan(R))
    vals = R[rows, cols].astype(np.float64)
    n_obs = len(rows)

    # RMSE before any training (random initialization)
    preds0 = np.sum(P[rows, :] * Q[cols, :], axis=1)
    rmse_start = np.sqrt(np.mean((vals - preds0) ** 2))

    prev_rmse = np.inf
    current_rmse = rmse_start
    last_delta = np.inf
    converged = False
    n_iter = 0
    history = []

    if verbose:
        print(f"SGD: k={k}, gamma={gamma}, lambda={lambda_}, "
              f"{n_obs} observed ratings, RMSE at init: {rmse_start:.4f}")

    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        order = rng.permutation(n_obs)
        _sgd_epoch(P, Q, rows, cols, vals, order, float(gamma), float(lambda_))

        # Training RMSE on all observed entries
        preds = np.sum(P[rows, :] * Q[cols, :], axis=1)
        current_rmse = np.sqrt(np.mean((vals - preds) ** 2))
        history.append(current_rmse)

        last_delta = abs(prev_rmse - current_rmse)
        if last_delta < tol:
            converged = True
            break

        prev_rmse = current_rmse

        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration:>4} - RMSE: {current_rmse:.4f} "
                  f"(change this epoch: {last_delta:.6f})")

    elapsed = time.perf_counter() - t0

    if verbose:
        if converged:
            print(f"  Converged at iteration {n_iter} "
                  f"(RMSE change {last_delta:.6f} < tol {tol})")
        else:
            print(f"  DID NOT converge after {n_iter} iterations "
                  f"(last RMSE change {last_delta:.6f} >= tol {tol})")
        drop = rmse_start - current_rmse
        print(f"  Train RMSE: {rmse_start:.4f} -> {current_rmse:.4f} "
              f"(drop {drop:.4f}, {100 * drop / rmse_start:.1f}%) | "
              f"time: {_fmt_time(elapsed)}")

    return {
        "P": P,
        "Q": Q,
        "R_hat": P @ Q.T if compute_rhat else None,
        "train_rmse": current_rmse,
        "rmse_start": rmse_start,
        "n_iter": n_iter,
        "converged": converged,
        "last_delta": last_delta,
        "history": history,
        "elapsed": elapsed,
    }


# ============================================================================
#     3) K-fold CV over a grid of (latent factors, lambda)
# ============================================================================

def kfold_cv_mf(R, k_folds=10, factors_range=None, lambda_range=None,
                gamma=0.01, max_iter=500, tol=1e-4, seed=1, verbose=True):
    """
    K-fold cross-validation for regularized matrix factorization.

    Every (k_factors, lambda) combination is evaluated on the SAME folds,
    so the comparison between combinations is fair.

    Returns
    -------
    pandas.DataFrame with columns: k_factors, lambda_, mean_rmse, sd_rmse,
    mean_train_rmse, mean_epochs, converged_folds, n_folds, time_sec,
    fold_rmse (list of per-fold test RMSEs).
    """
    if factors_range is None:
        factors_range = [2, 5, 8, 10]
    if lambda_range is None:
        lambda_range = [0.01, 0.05, 0.1, 0.2, 0.5]
    lambda_range = np.atleast_1d(lambda_range).tolist()

    if njit is not None:
        _warmup()

    observed_rows, observed_cols = np.where(~np.isnan(R))
    n_obs = len(observed_rows)

    # Random fold assignment: fold ids 1..k_folds repeated, then shuffled
    rng = np.random.default_rng(seed)
    fold_pattern = np.resize(np.arange(1, k_folds + 1), n_obs)
    fold_assignment = rng.permutation(fold_pattern)

    grid = list(itertools.product(factors_range, lambda_range))
    n_combos = len(grid)
    results = []
    best_so_far = np.inf
    t_start = time.perf_counter()

    if verbose:
        print("=" * 78)
        print(f"{k_folds}-fold CV | matrix {R.shape[0]} x {R.shape[1]} | "
              f"{n_obs} observed ratings ({100 * n_obs / R.size:.2f}% dense)")
        print(f"Factors : {', '.join(str(f) for f in factors_range)}")
        print(f"Lambdas : {', '.join(str(l) for l in lambda_range)}")
        print(f"gamma={gamma}, max_iter={max_iter}, tol={tol} | "
              f"{n_combos} combinations x {k_folds} folds = {n_combos * k_folds} fits")
        print(f"numba: {'ON' if njit is not None else 'OFF (slow! pip install numba)'}")
        print("=" * 78 + "\n")

    for combo_idx, (k, lam) in enumerate(grid, start=1):
        t_combo = time.perf_counter()

        if verbose:
            print(f"=== [{combo_idx}/{n_combos}] k = {k} factors, lambda = {lam} ===")

        fold_rmse = np.zeros(k_folds)
        fold_train = np.zeros(k_folds)
        fold_start = np.zeros(k_folds)
        fold_epochs = np.zeros(k_folds, dtype=int)
        fold_conv = np.zeros(k_folds, dtype=bool)

        for fold in range(1, k_folds + 1):
            test_mask = fold_assignment == fold
            test_rows = observed_rows[test_mask]
            test_cols = observed_cols[test_mask]

            # Train matrix: mask out the test entries
            R_train = R.copy()
            R_train[test_rows, test_cols] = np.nan

            model = sgd_aggarwal_reg(
                R_train, k=k, gamma=gamma, lambda_=lam,
                max_iter=max_iter, tol=tol, seed=seed,
                verbose=False, compute_rhat=False,
            )

            # Predict only the held-out entries
            predictions = np.sum(model["P"][test_rows, :] * model["Q"][test_cols, :], axis=1)
            actual = R[test_rows, test_cols]

            f = fold - 1
            fold_rmse[f] = np.sqrt(np.mean((actual - predictions) ** 2))
            fold_train[f] = model["train_rmse"]
            fold_start[f] = model["rmse_start"]
            fold_epochs[f] = model["n_iter"]
            fold_conv[f] = model["converged"]

            if verbose:
                status = "converged" if model["converged"] else "NOT converged"
                print(f"  Fold {fold:>2}/{k_folds} | test {fold_rmse[f]:.4f} | "
                      f"train {model['rmse_start']:.3f} -> {model['train_rmse']:.4f} | "
                      f"{model['n_iter']:>4} epochs, {status} | "
                      f"{_fmt_time(model['elapsed'])}")

        combo_time = time.perf_counter() - t_combo
        mean_rmse = fold_rmse.mean()
        sd_rmse = fold_rmse.std(ddof=1)

        results.append({
            "k_factors": k,
            "lambda_": lam,
            "mean_rmse": mean_rmse,
            "sd_rmse": sd_rmse,
            "mean_train_rmse": fold_train.mean(),
            "mean_epochs": fold_epochs.mean(),
            "converged_folds": int(fold_conv.sum()),
            "n_folds": k_folds,
            "time_sec": combo_time,
            "fold_rmse": fold_rmse.tolist(),
        })

        if verbose:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / combo_idx * (n_combos - combo_idx)
            gap = mean_rmse - fold_train.mean()
            print(f"  --> Test RMSE : mean {mean_rmse:.4f} (SD {sd_rmse:.4f})")
            print(f"  --> Train RMSE: mean {fold_train.mean():.4f} "
                  f"(started at {fold_start.mean():.3f}) | "
                  f"generalization gap (test - train): {gap:+.4f}")
            print(f"  --> Epochs    : mean {fold_epochs.mean():.0f} "
                  f"(min {fold_epochs.min()}, max {fold_epochs.max()}) | "
                  f"converged in {fold_conv.sum()}/{k_folds} folds")
            print(f"  --> Time      : {_fmt_time(combo_time)} this combo | "
                  f"{_fmt_time(elapsed)} elapsed | ETA ~{_fmt_time(eta)}")
            if mean_rmse < best_so_far:
                print("  --> *** NEW BEST so far ***")
                best_so_far = mean_rmse
            print()

    cv_results = pd.DataFrame(results)

    if verbose:
        print_cv_summary(cv_results, total_time=time.perf_counter() - t_start)

    return cv_results


def print_cv_summary(cv_results, total_time=None):
    """Final tables: all combinations ranked, and best lambda for each k."""
    df = cv_results.copy()
    df["gap"] = df["mean_rmse"] - df["mean_train_rmse"]
    df["converged"] = df["converged_folds"].astype(str) + "/" + df["n_folds"].astype(str)
    df["time"] = df["time_sec"].map(_fmt_time)
    df["mean_epochs"] = df["mean_epochs"].round(0).astype(int)

    cols = ["k_factors", "lambda_", "mean_rmse", "sd_rmse", "mean_train_rmse",
            "gap", "mean_epochs", "converged", "time"]
    fmt = lambda x: f"{x:.4f}"

    print("=" * 78)
    print("ALL COMBINATIONS (ranked by mean test RMSE)")
    print("=" * 78)
    print(df.sort_values("mean_rmse")[cols].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78)
    print("BEST LAMBDA FOR EACH k")
    print("=" * 78)
    best_per_k = df.loc[df.groupby("k_factors")["mean_rmse"].idxmin()]
    print(best_per_k[["k_factors", "lambda_", "mean_rmse", "sd_rmse", "gap"]]
          .to_string(index=False, float_format=fmt))

    best = df.loc[df["mean_rmse"].idxmin()]
    print(f"\nBEST OVERALL: k = {int(best['k_factors'])}, lambda = {best['lambda_']} "
          f"-> test RMSE {best['mean_rmse']:.4f} (SD {best['sd_rmse']:.4f})")

    n_bad = int((cv_results["converged_folds"] < cv_results["n_folds"]).sum())
    if n_bad:
        print(f"WARNING: {n_bad} combination(s) hit max_iter in at least one fold "
              f"without converging. Consider raising max_iter or tol.")
    if total_time is not None:
        print(f"Total CV time: {_fmt_time(total_time)}")
    print("=" * 78)


# ============================================================================
#                                   Main
# ============================================================================

if __name__ == "__main__":
    datos = pd.read_csv('Bases/ratings.csv')
    peliculas = pd.read_csv('Bases/movies.csv')

    # Optionally shrink the matrix: set N_USERS = 500 to sample 500 users
    N_USERS = None
    if N_USERS is not None:
        usuarios_muestra = datos['userId'].drop_duplicates().sample(n=N_USERS, random_state=42)
        datos = datos[datos['userId'].isin(usuarios_muestra)]

    df_rating = datos.pivot_table(index='userId', columns='movieId', values='rating')
    R = df_rating.to_numpy()

    # Grid to search. Larger k mostly adds cost; regularization is what
    # keeps big models from overfitting, so keep the lambda grid.
    factors_range = [2, 5, 8, 10, 20, 30, 50]
    lambda_range = [0.01]

    cv_results = kfold_cv_mf(
        R,
        k_folds=10,
        factors_range=factors_range,
        lambda_range=lambda_range,
        gamma=0.01,
        max_iter=1000,
    )

    # Refit on ALL observed ratings with the best combination
    best = cv_results.loc[cv_results["mean_rmse"].idxmin()]
    print("\nRefitting on all data with the best combination...")
    final_model = sgd_aggarwal_reg(
        R, k=int(best["k_factors"]), lambda_=float(best["lambda_"]),
        gamma=0.01, max_iter=1000, verbose=True,
    )
    R_hat = final_model["R_hat"]

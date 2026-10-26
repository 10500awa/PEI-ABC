"""
Biased matrix factorization via SGD with L2 regularization:

    r_hat(u, i) = mu + b_u + b_i + p_u . q_i

plus k-fold cross-validation over a GRID of (latent factors, lambda,
lambda_b, gamma), with progress reporting and two reference baselines
(global mean, and mu + b_u + b_i without latent factors).
"""

import itertools
import time

import numpy as np
import pandas as pd

# numba is optional, but makes the SGD loop ~50-100x faster.
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

def _sgd_epoch(P, Q, b_u, b_i, mu, rows, cols, vals, order,
               gamma, lambda_, lambda_b):
    """One pass over the observed ratings, in the given order (in place)."""
    for idx in order:
        i = rows[idx]
        j = cols[idx]
        e = vals[idx] - (mu + b_u[i] + b_i[j] + np.dot(P[i, :], Q[j, :]))

        # Biases: shrink towards 0 (i.e. towards the baseline mu)
        b_u[i] = b_u[i] + gamma * (e - lambda_b * b_u[i])
        b_i[j] = b_i[j] + gamma * (e - lambda_b * b_i[j])

        # Factors: sequential update (Q uses the already-updated P[i, :]).
        # For the canonical simultaneous update, save p_old = P[i, :].copy()
        # and use p_old in the Q line.
        P[i, :] = P[i, :] + gamma * (e * Q[j, :] - lambda_ * P[i, :])
        Q[j, :] = Q[j, :] + gamma * (e * P[i, :] - lambda_ * Q[j, :])


if njit is not None:
    _sgd_epoch = njit(cache=True)(_sgd_epoch)


def _predict(mu, b_u, b_i, P, Q, rows, cols, clip=None):
    """Full prediction mu + b_u + b_i + p.q for the given (row, col) pairs."""
    pred = mu + b_u[rows] + b_i[cols] + np.sum(P[rows, :] * Q[cols, :], axis=1)
    if clip is not None:
        pred = np.clip(pred, clip[0], clip[1])
    return pred


def _warmup():
    """Trigger numba compilation on a tiny matrix so it doesn't pollute timings."""
    R_small = np.array([[1.0, np.nan], [np.nan, 2.0]])
    sgd_aggarwal_reg(R_small, k=2, max_iter=1, verbose=False, compute_rhat=False)


# ============================================================================
#              2) Biased SGD matrix factorization
# ============================================================================

def sgd_aggarwal_reg(R, k=2, gamma=0.01, lambda_=0.1, lambda_b=None,
                     max_iter=1000, tol=1e-4, seed=1, verbose=True,
                     compute_rhat=True):
    """
    Biased matrix factorization via SGD with L2 regularization.

    Parameters
    ----------
    R : np.ndarray
        Ratings matrix with np.nan for missing entries.
    k : int
        Number of latent factors.
    gamma : float
        Learning rate (shared by factors and biases).
    lambda_ : float
        L2 regularization of the latent factors.
    lambda_b : float or None
        L2 regularization of the biases b_u, b_i. None -> same as lambda_.
    max_iter, tol, seed, verbose
        As before.
    compute_rhat : bool
        If True, also return the full prediction matrix
        mu + b_u + b_i + P @ Q.T (skip it in CV).

    Returns
    -------
    dict with keys: P, Q, b_u, b_i, mu, R_hat, train_rmse, rmse_start,
    n_iter, converged, last_delta, history, elapsed.
    """
    if lambda_b is None:
        lambda_b = lambda_

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape

    rows, cols = np.where(~np.isnan(R))
    vals = R[rows, cols].astype(np.float64)
    n_obs = len(rows)

    # Global mean of the ratings used for training (fixed during SGD)
    mu = float(vals.mean())

    P = rng.normal(0, 0.1, size=(n_users, k))
    Q = rng.normal(0, 0.1, size=(n_items, k))
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)

    # RMSE before any training: with biases at 0 this is ~ the global-mean RMSE
    preds0 = _predict(mu, b_u, b_i, P, Q, rows, cols)
    rmse_start = np.sqrt(np.mean((vals - preds0) ** 2))

    prev_rmse = np.inf
    current_rmse = rmse_start
    last_delta = np.inf
    converged = False
    n_iter = 0
    history = []

    if verbose:
        print(f"SGD: k={k}, gamma={gamma}, lambda={lambda_}, lambda_b={lambda_b}, "
              f"{n_obs} observed ratings, mu={mu:.4f}, RMSE at init: {rmse_start:.4f}")

    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        order = rng.permutation(n_obs)
        _sgd_epoch(P, Q, b_u, b_i, mu, rows, cols, vals, order,
                   float(gamma), float(lambda_), float(lambda_b))

        preds = _predict(mu, b_u, b_i, P, Q, rows, cols)
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

    R_hat = None
    if compute_rhat:
        R_hat = mu + b_u[:, None] + b_i[None, :] + P @ Q.T

    return {
        "P": P, "Q": Q, "b_u": b_u, "b_i": b_i, "mu": mu,
        "R_hat": R_hat,
        "train_rmse": current_rmse,
        "rmse_start": rmse_start,
        "n_iter": n_iter,
        "converged": converged,
        "last_delta": last_delta,
        "history": history,
        "elapsed": elapsed,
    }


# ============================================================================
#     3) Reference baselines (no latent factors)
# ============================================================================

def fit_bias_baseline(R, reg_i=10.0, reg_u=15.0, n_iter=10):
    """
    Regularized baseline estimates mu + b_u + b_i by alternating closed-form
    updates (Koren 2008 style). Used only as a reference line in the CV.
    """
    rows, cols = np.where(~np.isnan(R))
    vals = R[rows, cols]
    n_users, n_items = R.shape
    mu = vals.mean()
    cnt_u = np.bincount(rows, minlength=n_users)
    cnt_i = np.bincount(cols, minlength=n_items)
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)
    for _ in range(n_iter):
        b_i = np.bincount(cols, weights=vals - mu - b_u[rows], minlength=n_items) / (reg_i + cnt_i)
        b_u = np.bincount(rows, weights=vals - mu - b_i[cols], minlength=n_users) / (reg_u + cnt_u)
    return mu, b_u, b_i


# ============================================================================
#     4) K-fold CV over a grid of (latent factors, lambda, lambda_b, gamma)
# ============================================================================

def kfold_cv_mf(R, k_folds=10, factors_range=None, lambda_range=None,
                lambda_b_range=None, gamma_range=(0.01,), max_iter=500,
                tol=1e-4, seed=1, clip=None, verbose=True):
    """
    K-fold CV for biased, regularized matrix factorization.

    Every combination is evaluated on the SAME folds. Besides the model, two
    baselines are computed on the same folds: global mean, and
    mu + b_u + b_i (no factors). Cold-start entries (user or item with no
    training rating in that fold) are predicted with whatever biases exist
    (mu + b_u, mu + b_i or mu), so they are NOT excluded; their share is
    reported.

    lambda_b_range=None ties lambda_b to lambda_ in each combination.
    clip=(0.5, 5.0) clips predictions to the rating range before the RMSE.

    Returns a DataFrame; baselines are stored in cv_results.attrs["baselines"].
    """
    lambda_range = np.atleast_1d(lambda_range).tolist()
    gamma_range = np.atleast_1d(gamma_range).tolist()
    lambda_b_list = [None] if lambda_b_range is None else np.atleast_1d(lambda_b_range).tolist()

    if njit is not None:
        _warmup()

    observed_rows, observed_cols = np.where(~np.isnan(R))
    n_obs = len(observed_rows)

    rng = np.random.default_rng(seed)
    fold_pattern = np.resize(np.arange(1, k_folds + 1), n_obs)
    fold_assignment = rng.permutation(fold_pattern)

    # ---- Baselines and cold-start share, once per fold (independent of grid)
    base_mean = np.zeros(k_folds)
    base_bias = np.zeros(k_folds)
    cold_pct = np.zeros(k_folds)
    for fold in range(1, k_folds + 1):
        m = fold_assignment == fold
        tr, tc = observed_rows[m], observed_cols[m]
        R_train = R.copy()
        R_train[tr, tc] = np.nan
        actual = R[tr, tc]

        mu0 = np.nanmean(R_train)
        base_mean[fold - 1] = np.sqrt(np.mean((actual - mu0) ** 2))

        mu_b, bu, bi = fit_bias_baseline(R_train)
        pred_b = mu_b + bu[tr] + bi[tc]
        if clip is not None:
            pred_b = np.clip(pred_b, clip[0], clip[1])
        base_bias[fold - 1] = np.sqrt(np.mean((actual - pred_b) ** 2))

        n_u = (~np.isnan(R_train)).sum(axis=1)
        n_i = (~np.isnan(R_train)).sum(axis=0)
        cold_pct[fold - 1] = np.mean((n_u[tr] == 0) | (n_i[tc] == 0))

    baselines = {
        "global_mean_rmse": base_mean.mean(),
        "bias_only_rmse": base_bias.mean(),
        "cold_start_pct": 100 * cold_pct.mean(),
    }

    grid = list(itertools.product(factors_range, lambda_range, lambda_b_list, gamma_range))
    n_combos = len(grid)
    results = []
    best_so_far = np.inf
    t_start = time.perf_counter()

    if verbose:
        print("=" * 78)
        print(f"{k_folds}-fold CV | matrix {R.shape[0]} x {R.shape[1]} | "
              f"{n_obs} observed ratings ({100 * n_obs / R.size:.2f}% dense)")
        print(f"Factors  : {', '.join(str(f) for f in factors_range)}")
        print(f"Lambdas  : {', '.join(str(l) for l in lambda_range)}")
        print(f"Lambda_b : {'= lambda' if lambda_b_range is None else ', '.join(str(l) for l in lambda_b_list)}")
        print(f"Gammas   : {', '.join(str(g) for g in gamma_range)}")
        print(f"max_iter={max_iter}, tol={tol}, clip={clip} | "
              f"{n_combos} combinations x {k_folds} folds = {n_combos * k_folds} fits")
        print(f"numba: {'ON' if njit is not None else 'OFF (slow! pip install numba)'}")
        print("-" * 78)
        print("REFERENCE BASELINES (same folds, mean test RMSE)")
        print(f"  Global mean          : {baselines['global_mean_rmse']:.4f}")
        print(f"  mu + b_u + b_i       : {baselines['bias_only_rmse']:.4f}")
        print(f"  Cold-start entries   : {baselines['cold_start_pct']:.2f}% of test")
        print("=" * 78 + "\n")

    for combo_idx, (k, lam, lam_b, gam) in enumerate(grid, start=1):
        lam_b = lam if lam_b is None else lam_b
        t_combo = time.perf_counter()

        if verbose:
            print(f"=== [{combo_idx}/{n_combos}] k = {k}, lambda = {lam}, "
                  f"lambda_b = {lam_b}, gamma = {gam} ===")

        fold_rmse = np.zeros(k_folds)
        fold_train = np.zeros(k_folds)
        fold_start = np.zeros(k_folds)
        fold_epochs = np.zeros(k_folds, dtype=int)
        fold_conv = np.zeros(k_folds, dtype=bool)

        for fold in range(1, k_folds + 1):
            test_mask = fold_assignment == fold
            test_rows = observed_rows[test_mask]
            test_cols = observed_cols[test_mask]

            R_train = R.copy()
            R_train[test_rows, test_cols] = np.nan

            model = sgd_aggarwal_reg(
                R_train, k=k, gamma=gam, lambda_=lam, lambda_b=lam_b,
                max_iter=max_iter, tol=tol, seed=seed,
                verbose=False, compute_rhat=False,
            )

            # Full prediction (biases + factors) on held-out entries only
            predictions = _predict(model["mu"], model["b_u"], model["b_i"],
                                   model["P"], model["Q"],
                                   test_rows, test_cols, clip=clip)
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
            "lambda_b": lam_b,
            "gamma": gam,
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
            print(f"  --> Test RMSE : mean {mean_rmse:.4f} (SD {sd_rmse:.4f}) | "
                  f"vs mu+b_u+b_i: {mean_rmse - baselines['bias_only_rmse']:+.4f}")
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
    cv_results.attrs["baselines"] = baselines

    if verbose:
        print_cv_summary(cv_results, total_time=time.perf_counter() - t_start)

    return cv_results


def print_cv_summary(cv_results, total_time=None):
    """Final tables: all combinations ranked, and best combination for each k."""
    df = cv_results.copy()
    df["gap"] = df["mean_rmse"] - df["mean_train_rmse"]
    df["converged"] = df["converged_folds"].astype(str) + "/" + df["n_folds"].astype(str)
    df["time"] = df["time_sec"].map(_fmt_time)
    df["mean_epochs"] = df["mean_epochs"].round(0).astype(int)

    cols = ["k_factors", "lambda_", "lambda_b", "gamma", "mean_rmse", "sd_rmse",
            "mean_train_rmse", "gap", "mean_epochs", "converged", "time"]
    fmt = lambda x: f"{x:.4f}"

    print("=" * 78)
    print("ALL COMBINATIONS (ranked by mean test RMSE)")
    print("=" * 78)
    print(df.sort_values("mean_rmse")[cols].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78)
    print("BEST (lambda, lambda_b, gamma) FOR EACH k")
    print("=" * 78)
    best_per_k = df.loc[df.groupby("k_factors")["mean_rmse"].idxmin()]
    print(best_per_k[["k_factors", "lambda_", "lambda_b", "gamma", "mean_rmse",
                      "sd_rmse", "gap"]].to_string(index=False, float_format=fmt))

    best = df.loc[df["mean_rmse"].idxmin()]
    print(f"\nBEST OVERALL: k = {int(best['k_factors'])}, lambda = {best['lambda_']}, "
          f"lambda_b = {best['lambda_b']}, gamma = {best['gamma']} "
          f"-> test RMSE {best['mean_rmse']:.4f} (SD {best['sd_rmse']:.4f})")

    b = cv_results.attrs.get("baselines")
    if b:
        print(f"Baselines: global mean {b['global_mean_rmse']:.4f} | "
              f"mu+b_u+b_i {b['bias_only_rmse']:.4f}")
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

    factors_range = [10, 30]
    lambda_range = [0.05, 0.1, 0.5]
    lambda_b_range = None          # None -> lambda_b = lambda_ ; e.g. [0.02, 0.1] to tune it
    gamma_range = [0.005, 0.01]

    cv_results = kfold_cv_mf(
        R,
        k_folds=10,
        factors_range=factors_range,
        lambda_range=lambda_range,
        lambda_b_range=lambda_b_range,
        gamma_range=gamma_range,
        max_iter=1000,
        clip=None,                 # (0.5, 5.0) to clip predictions to the rating range
    )

    # Refit on ALL observed ratings with the best combination
    best = cv_results.loc[cv_results["mean_rmse"].idxmin()]
    final_model = sgd_aggarwal_reg(
        R, k=int(best["k_factors"]), lambda_=float(best["lambda_"]),
        lambda_b=float(best["lambda_b"]), gamma=float(best["gamma"]),
        max_iter=1000, verbose=True,
    )
    R_hat = final_model["R_hat"]   # already = mu + b_u + b_i + P @ Q.T

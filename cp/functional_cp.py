from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import det
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


@dataclass
class CPConfig:
    """
    Configuration for functional CP with PCA+GMM.

    Attributes
    ----------
    p_base : int
        Maximum number of retained PCA components (effective p = min(p_base, N, D)).
    K : int
        Number of mixture components in GMM fitted on PCA scores.
    alpha : float
        Miscoverage level (e.g., 0.05 => ~95% coverage).
    test_size : float
        Fraction of PCA scores used for calibration (remainder for GMM fit).
    random_state : int
        Global seed for PCA, train/validation split, and GMM.
    n_jobs : int
        Parallel workers used by precompute_all().
    backend : str
        Joblib backend for parallelization ("loky" recommended).
    """
    p_base: int = 3
    K: int = 4
    alpha: float = 0.05
    test_size: float = 0.30
    random_state: int = 0
    n_jobs: int = 1
    backend: str = "loky"


class PCAGMMResidualCP:
    """
    Functional CP model using PCA + GMM in score space.

    Input residual tensor can be:
      - (N, T_res, H, W)  -> image/field residuals
      - (N, T_res, D)     -> already flattened residuals

    Methods
    -------
    fit(residuals):
        Registers residual data (no heavy fitting up front).
    upper_at(t_idx):
        Computes g_upper(x) for residual time index t_idx.
    precompute_all():
        Computes g_upper for all t_idx in parallel and stacks results.

    Notes
    -----
    - For each t_idx:
        1) Flatten residuals across space: Yw in R^{N x D}
        2) PCA -> scores in R^{N x p}
        3) Split scores into Train/Calibration
        4) Fit GMM on Train, compute log-density on Calibration
        5) Set lambda = exp(q_log) with q_log the (1 - alpha)-quantile
        6) For each GMM component, derive an upper radius r_k in score space
        7) Map per-pixel upper envelope back to original space using PCA
    """

    def __init__(self, cfg: CPConfig):
        self.cfg = cfg
        self._residuals: Optional[np.ndarray] = None  # (N, T_res, H, W) or (N, T_res, D)
        self._HWD: Optional[Tuple[int, int, Optional[int]]] = None  # (H, W, D_if_flat)
        self._N: Optional[int] = None
        self._T_res: Optional[int] = None

    # ---------- public API ----------

    def fit(self, residuals: np.ndarray) -> None:
        """
        Register residual tensor.

        Parameters
        ----------
        residuals : np.ndarray
            Residuals with shape (N, T_res, H, W) or (N, T_res, D).
        """
        if residuals.ndim not in (3, 4):
            raise ValueError("residuals must have ndim 3 or 4: (N, T_res, D) or (N, T_res, H, W)")

        self._residuals = residuals.astype(np.float32, copy=False)

        if residuals.ndim == 4:
            N, T_res, H, W = residuals.shape
            self._HWD = (H, W, None)  # image mode
        else:
            N, T_res, D = residuals.shape
            self._HWD = (None, None, D)  # flat mode

        self._N = int(N)
        self._T_res = int(T_res)

    def upper_at(self, t_idx: int) -> np.ndarray:
        """
        Compute g_upper for a single residual index.

        Parameters
        ----------
        t_idx : int
            Residual time index in [0, T_res - 1].

        Returns
        -------
        np.ndarray
            g_upper with shape (H, W) if input was 4D, else shape (D,).
        """
        self._assert_fitted()
        return self._compute_upper_for_idx(t_idx)

    def precompute_all(self) -> np.ndarray:
        """
        Compute g_upper for all residual indices in parallel.

        Returns
        -------
        np.ndarray
            (T_res, H, W) if input was 4D, else (T_res, D).
        """
        self._assert_fitted()
        T_res = self._T_res
        assert T_res is not None

        results = Parallel(n_jobs=self.cfg.n_jobs, backend=self.cfg.backend)(
            delayed(self._compute_upper_for_idx)(t) for t in range(T_res)
        )

        # Stack along time dimension
        if self._is_image_mode():
            H, W, _ = self._extract_shapes()
            return np.asarray(results, dtype=np.float32).reshape(T_res, H, W)
        else:
            D = self._extract_shapes()[2]
            return np.asarray(results, dtype=np.float32).reshape(T_res, D)

    # ---------- internals ----------

    def _compute_upper_for_idx(self, t_idx: int) -> np.ndarray:
        """
        Core computation for a single t_idx:
        PCA -> GMM -> calibration quantile -> per-pixel upper envelope.
        """
        N, Yw, H, W, D = self._get_flat_view(t_idx)  # Yw: (N, D)

        # Effective PCA dimension
        p_eff = int(min(self.cfg.p_base, N, D))
        if p_eff <= 0:
            # No variance retained: return zeros (no residual uncertainty)
            return self._unflatten(np.zeros(D, dtype=np.float32), H, W)

        # PCA on residuals
        pca = PCA(n_components=p_eff, svd_solver="randomized", random_state=self.cfg.random_state)
        scores = pca.fit_transform(Yw)                         # (N, p_eff)
        A = pca.components_.astype(np.float64)                 # (p_eff, D) rows = PCs
        mean_vec = pca.mean_.astype(np.float64)                # (D,)

        # Train/Calibration split on scores
        # If N is too small for a split, fallback: use all as calibration
        if N >= 4:
            Xi_train, Xi_cal = train_test_split(
                scores,
                test_size=self.cfg.test_size,
                random_state=self.cfg.random_state,
            )
            # Edge case: split can produce empty train for very small N
            if len(Xi_train) < self.cfg.K + 1:
                Xi_train, Xi_cal = scores, scores
        else:
            Xi_train, Xi_cal = scores, scores

        # Fit GMM in score space
        K = min(self.cfg.K, max(1, len(Xi_train)))
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            random_state=self.cfg.random_state
        ).fit(Xi_train)

        # Calibration quantile in log-density
        logf_cal = gmm.score_samples(Xi_cal)                   # (n_cal,)
        q_log = np.quantile(logf_cal, 1.0 - self.cfg.alpha)    # (1 - alpha)-quantile
        lam = float(np.exp(q_log))

        pis = gmm.weights_                                     # (K,)
        mus = gmm.means_                                       # (K, p_eff)
        Sigmas = gmm.covariances_                              # (K, p_eff, p_eff)

        # Per-component radius in score space (ellipsoids)
        rks = np.zeros(K, dtype=np.float64)

        def _rk_sq(k: int) -> float:
            # lambda threshold translated into component-specific level set
            # tau accounts for mixture weights (simple heuristic consistent with user's approach)
            tau = lam / (K * pis[k])
            Sk = Sigmas[k]
            _det = det(Sk)
            if _det <= 0 or not np.isfinite(_det):
                # Regularize if degenerate
                Sk = Sk + 1e-8 * np.eye(Sk.shape[0])
                _det = det(Sk)
            # tau * (2π)^{p/2} * sqrt(det(Sk))  -> level-set scaling
            val = tau * ((2.0 * np.pi) ** (p_eff / 2.0)) * math.sqrt(max(_det, 0.0))
            if (not np.isfinite(val)) or val <= 0.0:
                return 0.0
            return max(0.0, -2.0 * math.log(val))

        for k in range(K):
            rks[k] = math.sqrt(_rk_sq(k))

        # Build pointwise envelope in original space
        AT = A.T                                            # (D, p_eff)
        upper = np.full((K, D), np.nan, dtype=np.float64)   # per-component upper
        for k in range(K):
            mu_k = mus[k]                                   # (p_eff,)
            rk = rks[k]
            Sk = Sigmas[k]                                  # (p_eff, p_eff)

            # Center of the ellipsoid mapped back
            center = AT @ mu_k                              # (D,)

            # Diagonal of A^T S_k A approximated via einsum
            AS = AT @ Sk                                    # (D, p_eff)
            quad_diag = np.einsum("dp,dp->d", AS, AT)       # (D,)
            quad_diag = np.clip(quad_diag, 0.0, np.inf)
            rad = rk * np.sqrt(quad_diag)                   # (D,)

            upper[k] = center + rad

        g_upper_vec = mean_vec + np.nanmax(upper, axis=0)   # (D,)
        return self._unflatten(g_upper_vec.astype(np.float32), H, W)

    # ---------- shape helpers ----------

    def _is_image_mode(self) -> bool:
        H, W, D = self._extract_shapes()
        return (H is not None) and (W is not None) and (D is None)

    def _extract_shapes(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        # Returns (H, W, D_if_flat)
        if self._HWD is None:
            return (None, None, None)
        return self._HWD

    def _get_flat_view(self, t_idx: int) -> Tuple[int, np.ndarray, Optional[int], Optional[int], int]:
        """
        Returns:
            N: number of episodes
            Yw: (N, D) flattened residuals at t_idx
            H, W: spatial dims if image mode, else None
            D: flattened dimension
        """
        if self._residuals is None:
            raise RuntimeError("Call fit() before _get_flat_view().")

        N = self._N
        T_res = self._T_res
        if N is None or T_res is None:
            raise RuntimeError("Invalid internal state: missing N or T_res.")
        if not (0 <= t_idx < T_res):
            raise IndexError(f"t_idx={t_idx} out of range [0, {T_res - 1}].")

        res = self._residuals
        if res.ndim == 4:
            # (N, T_res, H, W)
            _, _, H, W = res.shape
            D = H * W
            Yw = res[:, t_idx].reshape(N, D)
            return N, Yw, H, W, D
        else:
            # (N, T_res, D)
            D = res.shape[2]
            Yw = res[:, t_idx]
            return N, Yw, None, None, D

    def _unflatten(self, vec: np.ndarray, H: Optional[int], W: Optional[int]) -> np.ndarray:
        if H is not None and W is not None:
            return vec.reshape(H, W)
        return vec  # already flat

    def _assert_fitted(self) -> None:
        if self._residuals is None or self._N is None or self._T_res is None or self._HWD is None:
            raise RuntimeError("PCAGMMResidualCP is not fitted. Call fit(residuals) first.")
        


def compute_cp_upper_envelopes(
    residuals_train: np.ndarray,
    p_base: int,
    K: int,
    alpha: float,
    test_size: float,
    random_state: int,
    n_jobs: int,
    backend: str,
) -> np.ndarray:
    """
    Thin wrapper so callers don't need to instantiate the class manually.
    Returns g_upper_all_t with shape:
      - (T_res, H, W) if residuals were 4D (N, T_res, H, W)
      - (T_res, D)    if residuals were 3D (N, T_res, D)
    """
    cfg = CPConfig(
        p_base=p_base,
        K=K,
        alpha=alpha,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        backend=backend,
    )
    model = PCAGMMResidualCP(cfg)
    model.fit(residuals_train)
    return model.precompute_all()
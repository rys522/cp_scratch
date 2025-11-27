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
        Maximum number of retained PCA components.
        Effective components p = min(p_base, N, D).
    K : int
        Number of mixture components in GMM fitted on PCA scores.
    alpha : float
        Miscoverage level (e.g., 0.05 => 95% coverage).
    test_size : float
        Fraction of data used for calibration (remainder used for GMM fitting).
    random_state : int
        Global seed for reproducibility.
    n_jobs : int
        Number of parallel workers for precomputing envelopes.
    backend : str
        Joblib backend ("loky" recommended).
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
    Functional Conformal Prediction model using PCA + GMM.
    
    This class implements the score-space conformal prediction approach:
    1. Projects residuals to a lower-dimensional space via PCA.
    2. Calculates reconstruction error (epsilon) for validity.
    3. Fits a GMM on the scores to model density.
    4. Computes a conformal upper bound combining the GMM envelope and epsilon.

    Input shapes supported:
      - Image/Grid: (N, T_res, H, W)
      - Flat:       (N, T_res, D)
    """

    def __init__(self, cfg: CPConfig):
        self.cfg = cfg
        self._residuals: Optional[np.ndarray] = None
        # _HWD stores dimensions: (H, W, None) for images, (None, None, D) for flat vectors
        self._HWD: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None
        self._N: Optional[int] = None
        self._T_res: Optional[int] = None

    # ---------- Public API ----------

    def fit(self, residuals: np.ndarray) -> None:
        """
        Register residual tensor for calibration.
        Does not perform heavy computation until precompute_all() is called.

        Parameters
        ----------
        residuals : np.ndarray
            Shape (N, T_res, H, W) or (N, T_res, D).
        """
        if residuals.ndim not in (3, 4):
            raise ValueError("residuals must have ndim 3 or 4.")

        self._residuals = residuals.astype(np.float32, copy=False)

        if residuals.ndim == 4:
            N, T_res, H, W = residuals.shape
            self._HWD = (H, W, None)
        else:
            N, T_res, D = residuals.shape
            self._HWD = (None, None, D)

        self._N = int(N)
        self._T_res = int(T_res)

    def upper_at(self, t_idx: int) -> np.ndarray:
        """
        Compute the upper envelope g_upper(x) for a specific time step t_idx.
        """
        self._assert_fitted()
        return self._compute_upper_for_idx(t_idx)

    def precompute_all(self) -> np.ndarray:
        """
        Compute g_upper for all time steps in parallel.

        Returns
        -------
        np.ndarray
            Shape (T_res, H, W) or (T_res, D).
        """
        self._assert_fitted()
        T_res = self._T_res
        assert T_res is not None

        # 

        results = Parallel(n_jobs=self.cfg.n_jobs, backend=self.cfg.backend)(
            delayed(self._compute_upper_for_idx)(t) for t in range(T_res)
        )

        # Stack results along the time dimension
        if self._is_image_mode():
            H, W, _ = self._extract_shapes()
            return np.asarray(results, dtype=np.float32).reshape(T_res, H, W)
        else:
            D = self._extract_shapes()[2]
            return np.asarray(results, dtype=np.float32).reshape(T_res, D)

    # ---------- Internal Computation ----------

    def _compute_upper_for_idx(self, t_idx: int) -> np.ndarray:
        """
        Core logic for a single time step:
        1. PCA Projection & Reconstruction Error (Epsilon)
        2. GMM Density Estimation
        3. Conformal Calibration
        4. Reconstruction of Upper Envelope
        """
        N, Yw, H, W, D = self._get_flat_view(t_idx)  # Yw shape: (N, D)

        # Determine effective PCA components
        p_eff = int(min(self.cfg.p_base, N, D))
        if p_eff <= 0:
            # Trivial case: no variance to model
            return self._unflatten(np.zeros(D, dtype=np.float32), H, W)

        # --- Step 1: PCA Decomposition ---
        pca = PCA(n_components=p_eff, svd_solver="randomized", random_state=self.cfg.random_state)
        scores = pca.fit_transform(Yw)                         # (N, p_eff)
        A = pca.components_.astype(np.float64)                 # (p_eff, D)
        mean_vec = pca.mean_.astype(np.float64)                # (D,)

        # --- Step 2: Compute Reconstruction Error (Epsilon) ---
        # Theory: U(x) = Envelope_PCA(x) + Epsilon
        # Epsilon accounts for the information lost during dimension reduction.
        Y_recon = pca.inverse_transform(scores)
        # Calculate L_infty norm of error per sample: || y - y_recon ||_inf
        recon_errors = np.max(np.abs(Yw - Y_recon), axis=1)    # (N,)

        # --- Step 3: Split Data (Train / Calibration) ---
        # We split both the PCA scores and the reconstruction errors.
        if N >= 4:
            Xi_train, Xi_cal, err_train, err_cal = train_test_split(
                scores, recon_errors,
                test_size=self.cfg.test_size,
                random_state=self.cfg.random_state,
            )
            # Fallback for very small N if split leaves train set too small
            if len(Xi_train) < self.cfg.K + 1:
                Xi_train, Xi_cal = scores, scores
                err_train, err_cal = recon_errors, recon_errors
        else:
            # Not enough data to split, use all for both
            Xi_train, Xi_cal = scores, scores
            err_train, err_cal = recon_errors, recon_errors

        # --- Step 4: Calibrate Epsilon ---
        # Compute the (1 - alpha) quantile of the reconstruction errors
        epsilon = np.quantile(err_cal, 1.0 - self.cfg.alpha)

        # --- Step 5: GMM Fitting on Scores ---
        K = min(self.cfg.K, max(1, len(Xi_train)))
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            random_state=self.cfg.random_state
        ).fit(Xi_train)

        # --- Step 6: Calibrate Score Envelope ---
        # Compute log-likelihood of calibration data
        logf_cal = gmm.score_samples(Xi_cal)                   # (n_cal,)
        # Find threshold lambda using quantile of log-likelihoods
        q_log = np.quantile(logf_cal, 1.0 - self.cfg.alpha)
        lam = float(np.exp(q_log))

        pis = gmm.weights_                                     # (K,)
        mus = gmm.means_                                       # (K, p_eff)
        Sigmas = gmm.covariances_                              # (K, p_eff, p_eff)

        # Calculate radius r_k for each GMM component ellipsoid
        rks = np.zeros(K, dtype=np.float64)

        def _compute_rk_sq(k: int) -> float:
            # Effective threshold for component k
            tau = lam / (K * pis[k])
            Sk = Sigmas[k]
            _det = det(Sk)
            
            # Numerical stability check
            if _det <= 0 or not np.isfinite(_det):
                Sk = Sk + 1e-8 * np.eye(Sk.shape[0])
                _det = det(Sk)
            
            # Volume scaling factor
            val = tau * ((2.0 * np.pi) ** (p_eff / 2.0)) * math.sqrt(max(_det, 0.0))
            
            if (not np.isfinite(val)) or val <= 0.0:
                return 0.0
            return max(0.0, -2.0 * math.log(val))

        for k in range(K):
            rks[k] = math.sqrt(_compute_rk_sq(k))

        # --- Step 7: Map Envelope Back to Signal Space ---
        AT = A.T                                            # (D, p_eff)
        upper = np.full((K, D), np.nan, dtype=np.float64)   # Store upper bound per component
        
        for k in range(K):
            mu_k = mus[k]                                   # (p_eff,)
            rk = rks[k]
            Sk = Sigmas[k]                                  # (p_eff, p_eff)

            # Center of the ellipsoid mapped back
            center = AT @ mu_k                              # (D,)

            # Calculate the "width" of the ellipsoid in the direction of each pixel.
            # We want sqrt(e_i^T A^T S_k A e_i), which is the diagonal of A^T S_k A.
            AS = AT @ Sk                                    # (D, p_eff)
            quad_diag = np.einsum("dp,dp->d", AS, AT)       # Efficient diagonal computation
            quad_diag = np.clip(quad_diag, 0.0, np.inf)
            rad = rk * np.sqrt(quad_diag)                   # (D,)

            upper[k] = center + rad

        # Final Upper Bound: Mean + Max(Component Envelopes) + Epsilon
        g_upper_vec = mean_vec + np.nanmax(upper, axis=0) + epsilon
        
        return self._unflatten(g_upper_vec.astype(np.float32), H, W)

    # ---------- Helpers ----------

    def _is_image_mode(self) -> bool:
        H, W, D = self._extract_shapes()
        return (H is not None) and (W is not None) and (D is None)

    def _extract_shapes(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if self._HWD is None:
            return (None, None, None)
        return self._HWD

    def _get_flat_view(self, t_idx: int) -> Tuple[int, np.ndarray, Optional[int], Optional[int], int]:
        """
        Helper to flatten data regardless of input shape.
        Returns: (N, Y_flat, H, W, D)
        """
        if self._residuals is None:
            raise RuntimeError("Call fit() before accessing data.")

        N = self._N
        T_res = self._T_res
        if N is None or T_res is None:
            raise RuntimeError("Invalid internal state.")
        if not (0 <= t_idx < T_res):
            raise IndexError(f"t_idx={t_idx} out of range [0, {T_res - 1}].")

        res = self._residuals
        if res.ndim == 4:
            # Image mode: (N, T, H, W) -> Flatten spatial dims
            _, _, H, W = res.shape
            D = H * W
            Yw = res[:, t_idx].reshape(N, D)
            return N, Yw, H, W, D
        else:
            # Flat mode: (N, T, D)
            D = res.shape[2]
            Yw = res[:, t_idx]
            return N, Yw, None, None, D

    def _unflatten(self, vec: np.ndarray, H: Optional[int], W: Optional[int]) -> np.ndarray:
        """Restores spatial dimensions if necessary."""
        if H is not None and W is not None:
            return vec.reshape(H, W)
        return vec

    def _assert_fitted(self) -> None:
        if self._residuals is None or self._N is None:
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
    Functional interface for the PCAGMMResidualCP class.
    
    Parameters
    ----------
    residuals_train : np.ndarray
        Training residuals (N, T, H, W) or (N, T, D).
    p_base : int
        PCA components.
    K : int
        GMM components.
    alpha : float
        Miscoverage rate.
    
    Returns
    -------
    g_upper_all_t : np.ndarray
        Computed upper envelopes for all time steps.
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
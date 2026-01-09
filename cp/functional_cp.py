from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import slogdet
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from scipy.optimize import brentq


# ==============================================================================
# 0) Config / parameter containers
# ==============================================================================

@dataclass
class CPConfig:
    """
    Configuration for functional CP with PCA+GMM (+ LRW refinement).

    Notes
    -----
    - alpha_total is the overall miscoverage budget. If split_alpha=True,
      we use Bonferroni: alpha_eps = alpha_total/2, alpha_gmm = alpha_total/2
      to keep the final guarantee interpretation cleaner.
    """
    p_base: int = 3
    K: int = 4
    alpha_total: float = 0.05
    test_size: float = 0.30
    random_state: int = 0
    n_jobs: int = 1
    backend: str = "loky"

    # numerical stability
    cov_jitter: float = 1e-8


@dataclass
class CPStepParameters:
    """
    Offline parameter set Θ_i for the parametric functional upper envelope U_i(x)
    (cf. Eq. (24) and the paragraph "Offline parameterization" in the paper).

    We cache the full offline parameter set
        Θ_i = ( φ_i, ε_i, { μ_{k,i}, Σ_{k,i}, r_{k,i} }_{k=1}^K ),

    where:
    - φ_i        : projection basis (defining the operator Π_{p_i}),
                   represented in discretized form by `phi_basis`
                   (e.g., PCA components learned from residuals at step i);
    - ε_i        : projection/reconstruction slack satisfying
                   || S_{t+i|t} − Π_{p_i} S_{t+i|t} ||_∞ ≤ ε_i;
    - μ_{k,i}    : mean vector of the k-th Gaussian component in coefficient space;
    - Σ_{k,i}    : covariance matrix of the k-th Gaussian component in coefficient space;
    - r_{k,i}    : calibrated ellipsoidal radius associated with the k-th component.

    These parameters jointly define the conformal upper envelope
        U_i(x) = ε_i + max_k { μ_{k,i}^T φ_i(x)
                               + r_{k,i} ( φ_i(x)^T Σ_{k,i} φ_i(x) )^{1/2} }.
    """

    # horizon index i (time-to-go)
    t_idx: int

    # discretized projection basis φ_i:
    # shape (p_i, D), where each row corresponds to a basis function φ_{i,j}
    phi_basis: np.ndarray

    # projection slack ε_i (L_infty reconstruction error bound)
    epsilon: float

    # GMM means μ_{k,i} in coefficient space, shape (K, p_i)
    mus: np.ndarray

    # GMM covariances Σ_{k,i} in coefficient space, shape (K, p_i, p_i)
    sigmas: np.ndarray

    # refined ellipsoidal radii r_{k,i} for each mixture component, shape (K,)
    rks: np.ndarray

    # number of Gaussian mixture components K
    K: int

    # effective projection dimension p_i
    p_eff: int


# ==============================================================================
# 1) LRW refinement: QCQP-based eta(c) and delta_ks
#    (Lei–Rinaldo–Wasserman, Section 3.2 refinement idea)
# ==============================================================================

def _log_norm_const(Sigma: np.ndarray, p: int) -> float:
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Covariance not PD (slogdet sign<=0).")
    return 0.5 * (p * np.log(2.0 * np.pi) + logdet)

def _gauss_pdf(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    # stable log form
    p = mu.shape[0]
    diff = x - mu
    sol = np.linalg.solve(Sigma, diff)
    quad = float(diff.T @ sol)
    return float(np.exp(-0.5 * quad - _log_norm_const(Sigma, p)))

def _R2_from_c(c: float, Sigma_s: np.ndarray, p: int) -> float:
    """
    phi_s(x) >= c  <=>  (x-mu_s)^T Sigma_s^{-1} (x-mu_s) <= R^2(c)
    R^2(c) = -2 log( c (2π)^(p/2) |Sigma_s|^(1/2) ).
    """
    if c <= 0:
        return np.inf
    logZs = _log_norm_const(Sigma_s, p)
    val = -2.0 * (np.log(c) + logZs)
    # if c is larger than peak density, val<0; treat as infeasible => R2=0
    return float(max(0.0, val))

def _qcqp_min_mahalanobis(
    mu_k: np.ndarray,
    Sigma_k: np.ndarray,
    mu_s: np.ndarray,
    Sigma_s: np.ndarray,
    R2: float,
) -> Tuple[np.ndarray, float]:
    """
    Solve QCQP:
        minimize   (x-mu_k)^T Sigma_k^{-1} (x-mu_k)
        subject to (x-mu_s)^T Sigma_s^{-1} (x-mu_s) <= R2

    Use KKT with 1D Lagrange multiplier λ:
        x(λ) = (A + λB)^{-1}(A mu_k + λ B mu_s)
        A = Sigma_k^{-1}, B = Sigma_s^{-1}

    Returns: (x_star, dmin2)
    """
    p = mu_k.shape[0]

    # Precompute A and B once for this pair (p is small)
    A = np.linalg.solve(Sigma_k, np.eye(p))
    B = np.linalg.solve(Sigma_s, np.eye(p))

    # feasibility of x=mu_k
    diff0 = mu_k - mu_s
    g0 = float(diff0.T @ B @ diff0)
    if g0 <= R2 + 1e-12:
        return mu_k.copy(), 0.0

    def x_of_lam(lam: float) -> np.ndarray:
        M = A + lam * B
        rhs = A @ mu_k + lam * (B @ mu_s)
        return np.linalg.solve(M, rhs)

    def g(lam: float) -> float:
        x = x_of_lam(lam)
        d = x - mu_s
        return float(d.T @ B @ d - R2)

    # bracket λ: g(0)>0, g(∞)-> -R2 < 0
    lo, hi = 0.0, 1.0
    ghi = g(hi)
    it = 0
    while ghi > 0.0 and it < 60:
        hi *= 2.0
        ghi = g(hi)
        it += 1
    if ghi > 0.0:
        # fallback: conservative (rare)
        x = mu_s.copy()
        d = x - mu_k
        dmin2 = float(d.T @ A @ d)
        return x, dmin2

    lam_star = brentq(g, lo, hi, maxiter=200)
    x_star = x_of_lam(lam_star)
    dk = x_star - mu_k
    dmin2 = float(dk.T @ A @ dk)
    return x_star, dmin2

def _eta_of_c_QCQP(
    c: float,
    mu_k: np.ndarray,
    Sigma_k: np.ndarray,
    mu_s: np.ndarray,
    Sigma_s: np.ndarray,
) -> float:
    """
    η(c) = sup_x phi_k(x) s.t. phi_s(x) >= c.

    Convert constraint to ellipsoid via R^2(c), then solve QCQP.
    """
    p = mu_k.shape[0]
    # infeasible if c > peak_s
    peak_s = _gauss_pdf(mu_s, mu_s, Sigma_s)
    if c <= 0:
        return _gauss_pdf(mu_k, mu_k, Sigma_k)
    if c > peak_s:
        return 0.0

    R2 = _R2_from_c(c, Sigma_s, p)
    if not np.isfinite(R2):
        return _gauss_pdf(mu_k, mu_k, Sigma_k)

    x_star, dmin2 = _qcqp_min_mahalanobis(mu_k, Sigma_k, mu_s, Sigma_s, R2)
    # phi_k(x_star)
    logphi = -0.5 * dmin2 - _log_norm_const(Sigma_k, p)
    return float(np.exp(logphi))

def _delta_ks_LRW(
    pi_k: float, mu_k: np.ndarray, Sigma_k: np.ndarray,
    pi_s: float, mu_s: np.ndarray, Sigma_s: np.ndarray,
    eta_cache: Optional[Dict[Tuple[int, int, float], float]] = None,
    cache_key: Optional[Tuple[int, int]] = None,
) -> float:
    """
    δks = sup_x min(pi_k phi_k(x), pi_s phi_s(x)).

    Trivial case:
      if pi_s phi_s(mu_s) <= pi_k phi_k(mu_s),
      then δks = pi_s phi_s(mu_s).

    Otherwise solve for c* in (0, peak_s] such that
      pi_k * eta(c*) = pi_s * c*
    and δks = pi_s * c*.
    """
    peak_s = _gauss_pdf(mu_s, mu_s, Sigma_s)
    val_at_mu_s = pi_s * peak_s
    other_at_mu_s = pi_k * _gauss_pdf(mu_s, mu_k, Sigma_k)

    if val_at_mu_s <= other_at_mu_s:
        return float(val_at_mu_s)

    def eta(c: float) -> float:
        if eta_cache is None or cache_key is None:
            return _eta_of_c_QCQP(c, mu_k, Sigma_k, mu_s, Sigma_s)
        # quantize c a bit to make cache effective but safe
        cq = float(np.float64(c))
        key = (cache_key[0], cache_key[1], cq)
        if key in eta_cache:
            return eta_cache[key]
        v = _eta_of_c_QCQP(cq, mu_k, Sigma_k, mu_s, Sigma_s)
        eta_cache[key] = v
        return v

    def h(c: float) -> float:
        return pi_k * eta(c) - pi_s * c

    # bracket root on (0, peak_s]
    c_hi = peak_s
    c_lo = peak_s * 1e-12
    f_lo = h(c_lo)
    f_hi = h(c_hi)

    # should be f_lo>0, f_hi<0 in non-trivial case; handle corner numerics
    if f_lo < 0:
        return float(val_at_mu_s)
    if f_hi > 0:
        # rare numerical; fall back conservatively
        return float(val_at_mu_s)

    c_star = brentq(h, c_lo, c_hi, maxiter=200)
    return float(pi_s * c_star)

def compute_deltas_LRW_QCQP(
    pis: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    jitter: float = 1e-8,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Compute δ_k = sum_{s≠k} δ_{ks} using LRW QCQP refinement.
    δ_k depends only on mixture parameters, not on λ.
    """
    pis = np.asarray(pis, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    K = int(len(pis))
    p = int(mus.shape[1])

    # stabilize covariances
    sig = sigmas.copy()
    for k in range(K):
        sig[k] = sig[k] + jitter * np.eye(p)

    deltas = np.zeros(K, dtype=np.float64)
    eta_cache: Optional[Dict[Tuple[int, int, float], float]] = {} if use_cache else None

    for k in range(K):
        for s in range(K):
            if s == k:
                continue
            deltas[k] += _delta_ks_LRW(
                float(pis[k]), mus[k], sig[k],
                float(pis[s]), mus[s], sig[s],
                eta_cache=eta_cache,
                cache_key=(k, s) if use_cache else None,
            )
    return deltas.astype(np.float32)


# ==============================================================================
# 2) Main class: PCA + GMM + (LRW refined radii) + functional envelope
# ==============================================================================

class PCAGMMResidualCP:
    def __init__(self, cfg: CPConfig):
        self.cfg = cfg
        self._residuals: Optional[np.ndarray] = None
        self._HWD: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None
        self._N: Optional[int] = None
        self._T_res: Optional[int] = None

    def fit(self, residuals: np.ndarray) -> None:
        if residuals.ndim not in (3, 4):
            raise ValueError("residuals must have ndim 3 or 4.")
        self._residuals = residuals.astype(np.float32)
        if residuals.ndim == 4:
            N, T_res, H, W = residuals.shape
            self._HWD = (int(H), int(W), None)
        else:
            N, T_res, D = residuals.shape
            self._HWD = (None, None, int(D))
        self._N = int(N)
        self._T_res = int(T_res)

    # ---------- core extraction ----------

    def _extract_params_for_idx(self, t_idx: int) -> CPStepParameters:
        self._assert_fitted()
        N, Yw, H, W, D = self._get_flat_view(t_idx)
        p_eff = int(min(self.cfg.p_base, N, D))

        alpha_eps = self.cfg.alpha_total
        alpha_gmm = self.cfg.alpha_total

        # 1) PCA projection
        pca = PCA(n_components=p_eff, svd_solver="randomized", random_state=self.cfg.random_state)
        scores = pca.fit_transform(Yw)             # (N, p_eff)
        phi_basis = pca.components_                # (p_eff, D)

        # 2) reconstruction slack epsilon (L_infty across coordinates)
        Y_recon = pca.inverse_transform(scores)
        recon_errors = np.max(np.maximum(Y_recon - Yw, 0), axis=1)
        Xi_train, Xi_cal, _, err_cal = train_test_split(
            scores, recon_errors,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state
        )
        epsilon = float(np.quantile(err_cal, 1.0 - alpha_eps))

        # 3) Fit GMM and choose lambda (density superlevel) via calibration
        K_val = int(min(self.cfg.K, len(Xi_train)))
        gmm = GaussianMixture(
            n_components=K_val,
            covariance_type="full",
            random_state=self.cfg.random_state,
        ).fit(Xi_train)

        weighted_log_probs = gmm._estimate_weighted_log_prob(Xi_cal) # (N_cal, K)
        max_log_weighted_probs = np.max(weighted_log_probs, axis=1)  # (N_cal,)
        
        lam = float(np.exp(np.quantile(max_log_weighted_probs, alpha_gmm)))


        rks = np.zeros(K_val, dtype=np.float32)
        for k in range(K_val):
            pi_k = float(gmm.weights_[k])
            threshold_k = lam / pi_k 
            
            Sigma_k = gmm.covariances_[k] + self.cfg.cov_jitter * np.eye(p_eff)
            logZk = _log_norm_const(Sigma_k, p_eff)
            
            val_log = math.log(max(threshold_k, 1e-300)) + logZk
            rk_sq = max(0.0, -2.0 * val_log)
            rks[k] = float(math.sqrt(rk_sq))

        return CPStepParameters(
            t_idx=t_idx,
            phi_basis=phi_basis.astype(np.float32),
            epsilon=float(epsilon),
            mus=gmm.means_.astype(np.float32),
            sigmas=(gmm.covariances_ + self.cfg.cov_jitter * np.eye(p_eff)).astype(np.float32),
            rks=rks.astype(np.float32),
            K=K_val,
            p_eff=p_eff,
        )

    def _compute_upper_for_idx(self, t_idx: int) -> np.ndarray:
        """
        Precompute full-grid upper envelope for time index t_idx.
        """
        p = self._extract_params_for_idx(t_idx)
        D = int(p.phi_basis.shape[1])

        # In coefficient space: ellipsoid gives per-basis-direction support.
        # For discretized functions, we lift back via AT = components^T.
        AT = p.phi_basis.T  # (D, p_eff)

        upper_bounds = np.zeros((p.K, D), dtype=np.float32)
        for k in range(p.K):
            center = AT @ p.mus[k]  # (D,)

            # diag( A Σ A^T ) where A = AT, Σ is (p,p)
            AS = AT @ p.sigmas[k]   # (D,p)
            quad_diag = np.einsum("dp,dp->d", AS, AT)  # length D
            rad = p.rks[k] * np.sqrt(np.clip(quad_diag, 0.0, None))
            upper_bounds[k] = center + rad

        g_upper_vec = np.max(upper_bounds, axis=0) + p.epsilon

        H, W, _ = self._extract_shapes()
        return self._unflatten(g_upper_vec.astype(np.float32), H, W)

    # ---------- public APIs ----------

    def get_online_parameters_all(self) -> List[CPStepParameters]:
        self._assert_fitted()
        assert self._T_res is not None
        return Parallel(n_jobs=self.cfg.n_jobs, backend=self.cfg.backend)(
            delayed(self._extract_params_for_idx)(t) for t in range(self._T_res)
        )

    def precompute_all(self) -> np.ndarray:
        self._assert_fitted()
        assert self._T_res is not None
        results = Parallel(n_jobs=self.cfg.n_jobs, backend=self.cfg.backend)(
            delayed(self._compute_upper_for_idx)(t) for t in range(self._T_res)
        )
        return np.asarray(results, dtype=np.float32)

    # ---------- helpers ----------

    def _extract_shapes(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if self._HWD is None:
            return (None, None, None)
        return self._HWD

    def _get_flat_view(self, t_idx: int) -> Tuple[int, np.ndarray, Optional[int], Optional[int], int]:
        if self._residuals is None:
            raise RuntimeError("Call fit() before accessing data.")
        if self._N is None or self._T_res is None:
            raise RuntimeError("Invalid internal state.")
        if not (0 <= t_idx < self._T_res):
            raise IndexError(f"t_idx={t_idx} out of range [0, {self._T_res - 1}].")

        res = self._residuals
        N = self._N

        if res.ndim == 4:
            _, _, H, W = res.shape
            D = int(H * W)
            Yw = res[:, t_idx].reshape(N, D)
            return N, Yw, int(H), int(W), D
        else:
            D = int(res.shape[2])
            Yw = res[:, t_idx]
            return N, Yw, None, None, D

    def _unflatten(self, vec: np.ndarray, H: Optional[int], W: Optional[int]) -> np.ndarray:
        if H is not None and W is not None:
            return vec.reshape(H, W)
        return vec

    def _assert_fitted(self) -> None:
        if self._residuals is None or self._N is None or self._T_res is None:
            raise RuntimeError("PCAGMMResidualCP is not fitted. Call fit(residuals) first.")

    def precompute_all_from_params(self, params_list: List[CPStepParameters]) -> np.ndarray:
        """
        params_list: length T_res, each has ( phi_basis, mus, sigmas, rks, epsilon, ...)
        returns: (T_res, H, W) or (T_res, D)
        """
        self._assert_fitted()
        H, W, _ = self._extract_shapes()

        def one(p: CPStepParameters) -> np.ndarray:
            D = int(p.phi_basis.shape[1])
            AT = p.phi_basis.T  # (D, p_eff)

            upper_bounds = np.zeros((p.K, D), dtype=np.float32)
            for k in range(p.K):
                center = AT @ p.mus[k]          # (D,)
                AS = AT @ p.sigmas[k]           # (D,p)
                quad_diag = np.einsum("dp,dp->d", AS, AT)
                rad = float(p.rks[k]) * np.sqrt(np.clip(quad_diag, 0.0, None))
                upper_bounds[k] = center + rad

            g_upper_vec = np.max(upper_bounds, axis=0) + float(p.epsilon)
            return g_upper_vec.reshape(H, W) if (H is not None and W is not None) else g_upper_vec

        results = Parallel(n_jobs=self.cfg.n_jobs, backend=self.cfg.backend)(
            delayed(one)(p) for p in params_list
        )
        return np.asarray(results, dtype=np.float32)


# ==============================================================================
# 3) Functional interface
# ==============================================================================

def compute_cp_upper_envelopes(
    residuals_train: np.ndarray,
    p_base: int,
    K: int,
    alpha: float,
    test_size: float,
    random_state: int,
    n_jobs: int,
    backend: str,
    *,
    cov_jitter: float = 1e-8,
) -> np.ndarray:
    """
    residuals_train: (N,T,H,W) or (N,T,D)
    returns: (T,H,W) or (T,D)
    """
    cfg = CPConfig(
        p_base=p_base,
        K=K,
        alpha_total=alpha,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        backend=backend,
        cov_jitter=cov_jitter,
    )
    model = PCAGMMResidualCP(cfg)
    model.fit(residuals_train)

    params = model.get_online_parameters_all()

    g_upper = model.precompute_all_from_params(params)
    return g_upper

def get_envelopes_function(
    residuals_train: np.ndarray,
    p_base: int,
    K: int,
    alpha: float,
    test_size: float,
    random_state: int,
    n_jobs: int,
    backend: str,
    *,
    cov_jitter: float = 1e-8,
) -> List[CPStepParameters] :
    """
    residuals_train: (N,T,H,W) or (N,T,D)
    returns: (T,H,W) or (T,D), List[CPStepParameters]  (t_idx별 파라미터)
    """
    cfg = CPConfig(
        p_base=p_base,
        K=K,
        alpha_total=alpha,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        backend=backend,
        cov_jitter=cov_jitter,
    )
    model = PCAGMMResidualCP(cfg)
    model.fit(residuals_train)
    return model.get_online_parameters_all()

def get_envelopes_value_and_function(
    residuals_train: np.ndarray,
    p_base: int,
    K: int,
    alpha: float,
    test_size: float,
    random_state: int,
    n_jobs: int,
    backend: str,
    *,
    cov_jitter: float = 1e-8,
) -> Tuple[np.ndarray, List[CPStepParameters]]:
    cfg = CPConfig(
        p_base=p_base,
        K=K,
        alpha_total=alpha,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        backend=backend,
        cov_jitter=cov_jitter,
    )
    model = PCAGMMResidualCP(cfg)
    model.fit(residuals_train)

    params = model.get_online_parameters_all()

    g_upper = model.precompute_all_from_params(params)

    return g_upper, params
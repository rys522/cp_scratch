from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


@dataclass
class FunctionalCPConfig:
    alpha: float = 0.05
    test_size: float = 0.30
    random_state: int = 0
    n_jobs: int = 1
    backend: str = "loky"

    modulation: str = "sqrt"   # {"sqrt","sup"}
    score: str = "sqrt"         # {"sup","sqrt"}

    s_eps: float = 1e-6
    one_sided_positive: bool = True

    use_shared_basis: bool = True      # S_i를 공통 basis로 압축할지
    p_base: int = 16                   # basis dimension (공통 basis 크기)


@dataclass
class FunctionalCPParams:
    phi_basis: np.ndarray   # (p, D)  공통 basis (use_shared_basis=False면 None 가능)
    coeffs: np.ndarray      # (T, p)  horizon별 계수 a_i (use_shared_basis=False면 None 가능)
    rhos: np.ndarray        # (T,)
    grid_shape: tuple       # (nz,ny,nx) or (H,W) or (D,)
    x_star: np.ndarray      # (T, D)
    S_mean: Optional[np.ndarray] = None  # (D,)  (선택) PCA의 mean_ 저장용
    S_stack: Optional[np.ndarray] = None   


class FunctionalCP:
    def __init__(self, cfg: FunctionalCPConfig):
        self.cfg = cfg
        self._residuals: Optional[np.ndarray] = None
        self._N: Optional[int] = None
        self._T: Optional[int] = None
        self._grid_shape: Optional[Tuple[int, ...]] = None


    def fit(self, residuals: np.ndarray) -> None:
        if residuals.ndim not in (3, 4, 5):
            raise ValueError("residuals must have ndim 3, 4, or 5.")

        self._residuals = residuals.astype(np.float32)

        if residuals.ndim == 5:
            N, T, nz, ny, nx = residuals.shape
            self._grid_shape = (int(nz), int(ny), int(nx))
        elif residuals.ndim == 4:
            N, T, H, W = residuals.shape
            self._grid_shape = (int(H), int(W))
        else:
            N, T, D = residuals.shape
            self._grid_shape = (int(D),)

        self._N = int(N)
        self._T = int(T)

    def _assert_fitted(self):
        if self._residuals is None or self._N is None or self._T is None or self._grid_shape is None:
            raise RuntimeError("FunctionalCP is not fitted.")

    def _get_flat_view(self, t_idx: int):
        self._assert_fitted()
        res = self._residuals
        N = self._N
        assert N is not None

        if res.ndim == 5:
            _, _, nz, ny, nx = res.shape
            Y = res[:, t_idx].reshape(N, nz * ny * nx)
            return Y, (nz, ny, nx)
        if res.ndim == 4:
            _, _, H, W = res.shape
            Y = res[:, t_idx].reshape(N, H * W)
            return Y, (H, W)

        return res[:, t_idx], (res.shape[2],)

    def _unflatten(self, vec: np.ndarray, shape):
        if len(shape) == 1:
            return vec
        return vec.reshape(shape)

    def _quantile_higher(self, x: np.ndarray, q: float) -> float:
        """
        numpy 버전 호환: method="higher"가 없을 수도 있어서 안전하게 처리
        """
        x = np.asarray(x, dtype=float)
        x_sorted = np.sort(x)
        n = x_sorted.size
        if n == 0:
            return float("nan")
        # "higher": smallest value >= empirical quantile
        k = int(np.ceil(q * (n+1))) - 1
        k = int(np.clip(k, 0, n - 1))
        return float(x_sorted[k])

    def _extract_single(self, t_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        Y, _shape = self._get_flat_view(t_idx)  # (N, D)
        D = Y.shape[1]

        # split (Algorithm 1 step 1)
        Y_tr, Y_cal = train_test_split(
            Y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )

        # point prediction X* (여기선 mean residual)
        x_star = np.mean(Y_tr, axis=0).astype(np.float32)  # (D,)

        # calibration residuals relative to X*
        if self.cfg.one_sided_positive:
            E = np.maximum(Y_cal - x_star[None, :], 0.0)
        else:
            E = (Y_cal - x_star[None, :])

        # modulation S(x): eq (7) or (8)
        if self.cfg.modulation == "sup":
            S = np.max(np.abs(E), axis=0)
        elif self.cfg.modulation == "sqrt":
            S = np.sqrt(np.mean(E ** 2, axis=0))
        else:
            raise ValueError("Invalid modulation type.")

        S = np.maximum(S, self.cfg.s_eps).astype(np.float32)  # (D,)

        # conformity scores R_j: eq (9) or (10)
        if self.cfg.score == "sup":
            R = np.max(np.abs(E) / S[None, :], axis=1)
        elif self.cfg.score == "sqrt":
            R = np.sqrt(np.mean((E ** 2) / (S[None, :] ** 2), axis=1))
        else:
            raise ValueError("Invalid score type.")

        # quantile (Algorithm 1 step 4) - higher
        rho = self._quantile_higher(R, 1.0 - self.cfg.alpha)

        return x_star, S, float(rho)

    def extract_params_all(self) -> FunctionalCPParams:
        self._assert_fitted()
        T = self._T
        grid_shape = self._grid_shape
        assert T is not None and grid_shape is not None

        triples = Parallel(n_jobs=self.cfg.n_jobs, backend=self.cfg.backend)(
            delayed(self._extract_single)(t) for t in range(T)
        )
        x_star_list, S_list, rho_list = zip(*triples)

        x_star = np.stack(x_star_list, axis=0).astype(np.float32)  # (T, D)
        S_stack = np.stack(S_list, axis=0).astype(np.float32)      # (T, D)
        rhos = np.asarray(rho_list, dtype=np.float32)              # (T,)
        S_stack_keep = S_stack  # <- 저장

        # --- 공통 basis로 S_i 압축 ---
        if self.cfg.use_shared_basis:
            # PCA는 sample이 T개뿐이라 p_eff <= T
            p_eff = int(min(self.cfg.p_base, S_stack.shape[0], S_stack.shape[1]))
            pca = PCA(n_components=p_eff, svd_solver="randomized", random_state=self.cfg.random_state)
            coeffs = pca.fit_transform(S_stack).astype(np.float32)      # (T, p_eff)
            phi_basis = pca.components_.astype(np.float32)              # (p_eff, D)
            S_mean = pca.mean_.astype(np.float32)                       # (D,)
            S_stack_keep = None

        else:
            phi_basis = np.zeros((0, S_stack.shape[1]), dtype=np.float32)
            coeffs = np.zeros((T, 0), dtype=np.float32)
            S_mean = None
            S_stack_keep = S_stack  # <- 저장

        return FunctionalCPParams(
            phi_basis=phi_basis,
            coeffs=coeffs,
            rhos=rhos,
            grid_shape=grid_shape,
            x_star=x_star,
            S_mean=S_mean,
            S_stack=S_stack_keep,
        )
    
    def extract_params(self, *, rank: int = 16) -> FunctionalCPParams:
        """
        Algorithm 1로 각 t에 대해 (X*_t, S_t, rho_t)를 만든 뒤,
        S_t들을 (T,D)로 모아서 SVD로 rank 압축:
           S_t ≈ S_mean + coeffs[t] @ phi_basis
        """
        self._assert_fitted()
        T = int(self._T)
        # D는 flatten 기준
        Y0, shape0 = self._get_flat_view(0)
        D = int(Y0.shape[1])

        S_all = np.zeros((T, D), dtype=np.float32)
        X_star_all = np.zeros((T, D), dtype=np.float32)
        rhos = np.zeros((T,), dtype=np.float32)

        for t_idx in range(T):
            Y, shape = self._get_flat_view(t_idx)
            if shape != shape0:
                raise ValueError("grid_shape mismatch across horizons.")

            # split
            Y_tr, Y_cal = train_test_split(
                Y,
                test_size=self.cfg.test_size,
                random_state=self.cfg.random_state,
            )

            # X* (mean predictor)
            X_star = np.mean(Y_tr, axis=0).astype(np.float32)
            X_star_all[t_idx] = X_star

            # E = residuals wrt X*
            if self.cfg.one_sided_positive:
                E = np.maximum(Y_cal - X_star[None, :], 0.0)
            else:
                E = (Y_cal - X_star[None, :]).astype(np.float32)

            # S(x): modulation
            if self.cfg.modulation == "sup":
                S = np.max(np.abs(E), axis=0)
            elif self.cfg.modulation == "sqrt":
                S = np.sqrt(np.mean(E ** 2, axis=0))
            else:
                raise ValueError("Invalid modulation type.")
            S = np.maximum(S, float(self.cfg.s_eps)).astype(np.float32)
            S_all[t_idx] = S

            # R_j: conformity scores
            if self.cfg.score == "sup":
                R = np.max(np.abs(E) / S[None, :], axis=1)
            elif self.cfg.score == "sqrt":
                R = np.sqrt(np.mean((E ** 2) / (S[None, :] ** 2), axis=1))
            else:
                raise ValueError("Invalid score type.")

            # rho_t
            rho = self._quantile_higher(R, 1.0 - self.cfg.alpha)
            rhos[t_idx] = rho

        # ---- low-rank compression of S_all ----
        S_mean = np.mean(S_all, axis=0).astype(np.float32)
        A = (S_all - S_mean[None, :]).astype(np.float32)  # (T,D)

        # SVD: A = U S Vt
        U_svd, S_svd, Vt = np.linalg.svd(A, full_matrices=False)
        p_eff = int(min(rank, Vt.shape[0]))
        phi_basis = Vt[:p_eff, :].astype(np.float32)                       # (p,D)
        coeffs = (U_svd[:, :p_eff] * S_svd[:p_eff]).astype(np.float32)     # (T,p)

        return FunctionalCPParams(
            phi_basis=phi_basis,
            coeffs=coeffs,
            rhos=rhos,
            grid_shape=shape0,
            S_mean=S_mean,
            x_star=X_star_all,
        )

    # ------------------------------------------------------
    # (선택) params로부터 U_i를 복원하는 유틸
    # ------------------------------------------------------
    @staticmethod
    def reconstruct_S_from_params(params: FunctionalCPParams, t_idx: int) -> np.ndarray:
        """
        returns S_i as flat vector (D,)
        """
        if params.phi_basis.shape[0] == 0:
            raise ValueError("No basis stored (use_shared_basis=False). Store S separately if needed.")
        # S_i ≈ mean + a_i @ phi_basis
        S_flat = params.S_mean + (params.coeffs[t_idx] @ params.phi_basis)
        return S_flat.astype(np.float32)

    @staticmethod
    def reconstruct_U_from_params(params: FunctionalCPParams, t_idx: int) -> np.ndarray:
        """
        returns U_i in grid shape
        """
        S_flat = FunctionalCP.reconstruct_S_from_params(params, t_idx)
        U_flat = params.rhos[t_idx] * S_flat
        return U_flat.reshape(params.grid_shape).astype(np.float32)
import numpy as np
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from numpy.linalg import cholesky, det, inv
from math import pi
import matplotlib.pyplot as plt
import cvxpy as cp

K = 5
p = 3

# 0) Load example dataset: Berkeley Growth
ds = skfda.datasets.fetch_growth()
fd = ds['data']                # FunctionalData object
grid = fd.grid_points[0]       # grid points (ages)

# 1) FPCA: project functions into p-dimensional score space
fpca = FPCA(n_components=p, centering=True)
scores = fpca.fit_transform(fd)            # shape: (n_samples, p)
phi_fd = fpca.components_                  # eigenfunctions (basis functions)
mu_fd  = fpca.mean_                        # mean function

# 2) Train/calibration split
Xi_train, Xi_cal = train_test_split(scores, test_size=0.3, random_state=0) # split train/calibration set scores(cofficients)

# 3) Fit a Gaussian Mixture Model (EM algorithm under the hood)
gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=0)
gmm.fit(Xi_train)

# 4) Conformal threshold λ
#    Nonconformity score: s(x) = -f(x), so larger density → smaller score
#    λ chosen as the quantile of calibration scores
alpha = 0.1
logf_cal = gmm.score_samples(Xi_cal)   # log f(x)
s_cal = -np.exp(logf_cal)              # scores = -f(x)
lam = -np.quantile(s_cal, 1 - alpha)   # conformal threshold λ

# 5) Ellipsoid parameters for each mixture component
#    T_{n,k} = {ξ : φ_k(ξ; μ_k, Σ_k) ≥ λ/(K π_k)}
pis = gmm.weights_          # (K,)
mus = gmm.means_            # (K,p)
Sigmas = gmm.covariances_   # (K,p,p)


def gaussian_density(x, mu, Sigma):
    p = mu.shape[0]
    diff = x - mu
    try:
        invS = inv(Sigma)
        detS = det(Sigma)
    except np.linalg.LinAlgError:
        Sigma = Sigma + 1e-8 * np.eye(mu.shape[0])
        invS = inv(Sigma)
        detS = det(Sigma)
    quad = diff @ invS @ diff
    norm_const = 1.0 / (((2*pi)**(p/2)) * (np.sqrt(detS)))
    return norm_const * np.exp(-0.5 * quad)


# --- QCQP-based delta_ks using CVXPY ------------------------------------------

def R2_from_c(Sigma, c, p):
    # R^2(c) = -2 log( c (2π)^(p/2) |Σ|^(1/2) )
    if c <= 0:
        return np.inf
    detS = det(Sigma)
    val = c * ((2*np.pi)**(p/2)) * (np.sqrt(detS))
    if val <= 0:
        return np.inf
    return max(0.0, -2.0 * np.log(val))

def eta_of_c_QCQP(c, mu_k, Sigma_k, mu_s, Sigma_s):
    """
    Solve: min (x - mu_k)^T Sigma_k^{-1} (x - mu_k)
           s.t. (x - mu_s)^T Sigma_s^{-1} (x - mu_s) <= R^2(c)
    Return: eta(c) = const_k * exp(-0.5 * dmin2)
    """
    p = mu_k.shape[0]
    R2 = R2_from_c(Sigma_s, c, p)
    if not np.isfinite(R2):  # infeasible threshold
        return 0.0

    # Variables
    x = cp.Variable(p)

    # Precompute inverses (regularize if needed)
    try:
        invSk = np.linalg.inv(Sigma_k)
        invSs = np.linalg.inv(Sigma_s)
    except np.linalg.LinAlgError:
        Sigma_k = Sigma_k + 1e-8*np.eye(p)
        Sigma_s = Sigma_s + 1e-8*np.eye(p)
        invSk = np.linalg.inv(Sigma_k)
        invSs = np.linalg.inv(Sigma_s)

    # Objective (quadratic, convex)
    obj = cp.quad_form(x - mu_k, invSk)

    # Constraint (ellipsoidal)
    constr = [cp.quad_form(x - mu_s, invSs) <= R2]

    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve(solver=cp.SCS, verbose=False)  # ECOS도 가능: solver=cp.ECOS

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return 0.0

    dmin2 = obj.value
    if dmin2 is None or not np.isfinite(dmin2):
        return 0.0

    # eta(c) = (2π)^(-p/2) |Σ_k|^(-1/2) * exp(-0.5 * dmin2)
    detSk = det(Sigma_k)
    const_k = 1.0 / (((2*np.pi)**(p/2)) * (np.sqrt(detSk)))
    return float(const_k * np.exp(-0.5 * dmin2))

def delta_ks_qcqp(k, s, pis, mus, Sigmas, tol=1e-6, max_iter=40):
    """
    Find delta_ks = sup_x min(π_k φ_k(x), π_s φ_s(x))
    by solving π_k c* = π_s η(c*), with bisection on c in (0, φ_s(μ_s)].
    """
    p = mus.shape[1]
    mu_k, mu_s = mus[k], mus[s]
    Sig_k, Sig_s = Sigmas[k], Sigmas[s]

    # upper bound for c is max density of component s at its mean
    const_s = 1.0 / (((2*np.pi)**(p/2)) * (np.sqrt(det(Sig_s))))
    c_lo, c_hi = 0.0, const_s  # open at 0

    # Quick check: trivial case (often in the paper)
    # If at μ_s the weighted s-density already <= weighted k-density at its own mean,
    # δ_ks is simply min(π_k φ_k(μ_k), π_s φ_s(μ_s)) which equals π_s φ_s(μ_s) at tie point.
    # We'll still run bisection; but you can short-circuit if desired.

    # Bisection on c: solve π_k c = π_s η(c)
    for _ in range(max_iter):
        c_mid = 0.5 * (c_lo + c_hi)
        eta_mid = eta_of_c_QCQP(c_mid, mu_k, Sig_k, mu_s, Sig_s)
        lhs = pis[k] * c_mid
        rhs = pis[s] * eta_mid

        if not np.isfinite(eta_mid):
            # if failed, shrink interval
            c_hi = c_mid
            continue

        if abs(lhs - rhs) <= tol * max(1.0, rhs, lhs):
            c_star = c_mid
            break

        if lhs < rhs:
            # increase c to balance
            c_lo = c_mid
        else:
            c_hi = c_mid
    else:
        c_star = 0.5 * (c_lo + c_hi)

    # δ_ks = π_k * c*
    return float(pis[k] * c_star)

def delta_k_qcqp(k, pis, mus, Sigmas):
    vals = []
    K = len(pis)
    for s in range(K):
        if s == k: 
            continue
        vals.append(delta_ks_qcqp(k, s, pis, mus, Sigmas))
    return max(vals) if vals else 0.0

def rk_squared(k, lam):
    # Compute the ellipsoid radius squared (r_k^2) for the k-th Gaussian component
    # given the conformal threshold λ.
    
    # Step 1: Define the density threshold τ for component k.
    # This comes from the outer bound condition:
    #   φ_k(ξ; μ_k, Σ_k) >= λ / (K * π_k)
    tau = lam / (K * pis[k])  # λ / (K π_k)
    
    # Step 2: Access the covariance matrix of component k.
    Sigma = Sigmas[k]
    
    # Step 3: Rearrange the Gaussian density inequality:
    #   φ(ξ; μ, Σ) >= τ
    # ↔ exp(-0.5 * (ξ-μ)^T Σ^{-1} (ξ-μ)) >= τ * (2π)^(p/2) * |Σ|^(1/2)
    #
    # Define the RHS constant:
    val = tau * ((2*np.pi)**(p/2)) * (np.sqrt(det(Sigma)))  
    # = τ * (2π)^(p/2) * |Σ|^(1/2)
    
    # Step 4: Take logs and rearrange:
    #   (ξ-μ)^T Σ^{-1} (ξ-μ) <= -2 log(val)
    #
    # This gives an ellipsoid centered at μ with shape Σ
    # and radius squared r_k^2 = -2 log(val).
    #
    # Step 5: Numerical safeguard:
    # If val > 1, log(val) > 0 → RHS becomes negative,
    # meaning the inequality describes an empty set.
    # In that case, return 0.0 to avoid negative radius squared.
    return max(0.0, -2.0 * np.log(val))

# Only valid if all delta_k < lam (as per paper)

delta_ks = np.array([delta_k_qcqp(k, pis, mus, Sigmas) for k in range(K)])
use_tight = np.all(delta_ks < lam)

# Compute tight radii: tau_tight = max( (lam - delta_k)/pi_k, 0 )
def rk_tight_squared(k, lam, delta_k):
    tau = (lam - delta_k) / pis[k]
    if tau <= 0.0:
        return 0.0                     
    Sigma = Sigmas[k]
    val = tau * ((2*np.pi)**(p/2)) * (np.sqrt(det(Sigma)))
    if not np.isfinite(val) or val <= 0.0:
        return 0.0
    return max(0.0, -2.0*np.log(val))

rks_tight = (np.array([np.sqrt(rk_tight_squared(k, lam, delta_ks[k])) for k in range(K)])
             if use_tight else np.zeros(K))

rks = np.array([np.sqrt(rk_squared(k, lam)) for k in range(K)])
Ls = np.array([cholesky(Sigmas[k]) for k in range(K)])  # Σ^{1/2} via Cholesky

# 6) Compute prediction band on the grid
#    For each t: ell_k(t) = a^T μ_k - r_k ||Σ^{1/2} a||,
#                 upp_k(t) = a^T μ_k + r_k ||Σ^{1/2} a||
phi_mat = np.vstack([phi_fd[i](grid).squeeze() for i in range(p)]) # phi_mat = phi(t) = (phi_1(t),...,phi_p(t))^T, shape: (p,T)

def band_on_grid(rks):
    ell = np.full((K, len(grid)), np.nan)
    upp = np.full((K, len(grid)), np.nan)
    for ti, t in enumerate(grid):
        a = phi_mat[:, ti]  # vector of eigenfunctions evaluated at t
        for k in range(K):
            mu_k = mus[k]
            rk = rks[k]
            La = Ls[k] @ a         # Σ^{1/2} a
            rad = rk * np.linalg.norm(La)
            if not np.isfinite(rad):
                rad = 0.0
            center = mu_k @ a
            #    For each t: ell_k(t) = a^T μ_k - r_k ||Σ^{1/2} a||,
            #                upp_k(t) = a^T μ_k + r_k ||Σ^{1/2} a||    
            ell[k, ti] = center - rad
            upp[k, ti] = center + rad
    # Envelope: union of intervals approximated by min lower and max upper
    ell_env = ell.min(axis=0)
    upp_env = upp.max(axis=0)
    return ell, upp, ell_env, upp_env

ell, upp, ell_env, upp_env = band_on_grid(rks)

# 7) Add back the mean function to reconstruct band for X(t)
mu_vals = mu_fd(grid).squeeze()
lower_env = mu_vals + ell_env
upper_env = mu_vals + upp_env

ell_t, upp_t, ell_env_t, upp_env_t = band_on_grid(rks_tight)
lower_env_tight = mu_vals + ell_env_t
upper_env_tight = mu_vals + upp_env_t

# 8) Plot the results

Y = fd.data_matrix[..., 0]           # (n_samples, T)
T = np.asarray(grid).ravel()         # (T,)

plt.figure(figsize=(8, 5))

for i in range(min(50, Y.shape[0])):     
    plt.plot(T, Y[i, :], linewidth=0.7, alpha=0.35, zorder=1)

# loose band (blue)
plt.fill_between(T, lower_env, upper_env, color="skyblue", alpha=0.4, label="Prediction band (loose)")

# tight band (orange edges + transparent fill)
plt.fill_between(T, lower_env_tight, upper_env_tight, color="orange", alpha=0.2, label="Prediction band (tight)")
plt.plot(T, lower_env_tight, color="orange", linewidth=1.5)
plt.plot(T, upper_env_tight, color="orange", linewidth=1.5)


plt.xlabel("t")
plt.ylabel("X(t)")
plt.title("Functional Data with Conformal Prediction Band")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# --- After fitting FPCA ---
var_ratio = np.asarray(fpca.explained_variance_ratio_)  # shape: (p,)
cum_ratio = np.cumsum(var_ratio)

print("Explained variance ratio per PC:", np.round(var_ratio, 4))
print("Cumulative explained variance:", np.round(cum_ratio, 4))

# Example thresholds
thr = 0.95
p_sufficient = int(np.searchsorted(cum_ratio, thr) + 1)
print(f"p sufficient for {thr*100:.0f}% cumulative variance:", p_sufficient)

print("λ =", lam)
print("δ_k (QCQP) =", np.round(delta_ks, 6))
print("rks (loose) =", np.round(rks, 3))
print("rks_tight =", np.round(rks_tight, 3))
"""
AM205 Project: Solving Lasso Problem via ADMM
Authors: Donglai Wu, Ori Wang
Date: December 9, 2025

This script implements the Alternating Direction Method of Multipliers (ADMM)
to solve the Lasso regression problem and compares its performance against
a standard Subgradient Descent baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_synthetic_lasso(n=200, p=500, s=20, noise_std=0.1, seed=42): # Set seed for reproducibility
    """
    Generates synthetic data matching the original notebook exactly.
    """
    # Use the same generator as the notebook
    rng = np.random.default_rng(seed)

    # Design matrix
    A = rng.normal(size=(n, p))

    # True sparse vector
    x_true = np.zeros(p)
    support = rng.choice(p, size=s, replace=False)
    x_true[support] = rng.normal(size=s)

    # Noise
    noise = noise_std * rng.normal(size=n)
    b = A @ x_true + noise

    return A, b, x_true

def lasso_objective(A, b, x, lam):
    """Calculates the Lasso objective function: 0.5 * ||Ax - b||^2 + lam * ||x||_1"""
    return 0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.linalg.norm(x, 1)

def soft_threshold(v, tau):
    """Proximal operator for the L1 norm."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

def admm_lasso(A, b, lam, rho=1.0, max_iter=1000, abs_tol=1e-4, verbose=True):
    """
    Solves Lasso using ADMM.
    
    Args:
        A: Design matrix
        b: Response vector
        lam: Regularization parameter
        rho: Augmented Lagrangian penalty parameter
    """
    n, p = A.shape
    
    # Pre-compute Cholesky or matrix inverse components (A^T A + rho I)
    AtA = A.T @ A
    Atb = A.T @ b
    # For p=500, a direct inverse is fast enough. For larger p, use Cholesky.
    LHS = AtA + rho * np.eye(p)
    
    x = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    
    history = {"objval": [], "r_norm": [], "s_norm": []}
    
    print("Starting ADMM...")
    for k in range(max_iter):
        # 1. x-update: Smooth quadratic minimization
        rhs = Atb + rho * (z - u)
        x = np.linalg.solve(LHS, rhs)
        
        # 2. z-update: Soft thresholding (handles sparsity)
        x_hat = x + u
        z_old = z.copy()
        z = soft_threshold(x_hat, lam / rho)
        
        # 3. u-update: Dual variable update
        u = u + x - z
        
        # Residuals
        r = x - z # Primal residual
        s = rho * (z - z_old) # Dual residual
        
        # Log history
        obj = lasso_objective(A, b, x, lam)
        history["objval"].append(obj)
        history["r_norm"].append(np.linalg.norm(r))
        history["s_norm"].append(np.linalg.norm(s))
        
        # Stopping criteria
        if np.linalg.norm(r) < abs_tol and np.linalg.norm(s) < abs_tol:
            if verbose:
                print(f"ADMM converged at iteration {k}")
            break
            
    return z, history

def subgradient_lasso(A, b, lam, step_size=1e-4, max_iter=5000):
    """
    Baseline solver: Subgradient Descent.
    Note: Requires small step size to converge due to non-smoothness.
    """
    n, p = A.shape
    x = np.zeros(p)
    obj_hist = []
    
    print("Starting Subgradient Descent...")
    for k in range(max_iter):
        grad_smooth = A.T @ (A @ x - b)
        subgrad_l1 = lam * np.sign(x)
        grad = grad_smooth + subgrad_l1
        
        x = x - step_size * grad
        obj_hist.append(lasso_objective(A, b, x, lam))
        
    return x, obj_hist

def plot_results(x_true, x_admm, x_gd, hist_admm, hist_gd):
    """Generates comparison plots."""
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    # 1. Convergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(hist_admm['objval'], label='ADMM', linewidth=2, color='blue')
    plt.plot(hist_gd, label='Subgradient Descent', linewidth=2, color='orange', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Convergence Comparison: ADMM vs Subgradient Descent')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('convergence_plot.png')
    plt.show()

    # 2. Coefficient Recovery
    def rel_err(x, true): return np.linalg.norm(x - true) / np.linalg.norm(true)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(x_true, x_admm, alpha=0.6, color='blue', edgecolor='k')
    plt.plot([min(x_true), max(x_true)], [min(x_true), max(x_true)], 'r--')
    plt.title(f'ADMM Recovery (Rel Err: {rel_err(x_admm, x_true):.3f})')
    plt.xlabel('True Coefficients'); plt.ylabel('Estimated')

    plt.subplot(1, 2, 2)
    plt.scatter(x_true, x_gd, alpha=0.6, color='orange', edgecolor='k')
    plt.plot([min(x_true), max(x_true)], [min(x_true), max(x_true)], 'r--')
    plt.title(f'GD Recovery (Rel Err: {rel_err(x_gd, x_true):.3f})')
    plt.xlabel('True Coefficients'); plt.ylabel('Estimated')
    
    plt.tight_layout()
    plt.savefig('recovery_plot.png')
    plt.show()

    # 3. Sparsity Pattern
    plt.figure(figsize=(12, 4))
    idx = np.arange(100)
    plt.bar(idx, x_true[idx], color='green', alpha=0.5, label='True')
    plt.plot(idx, x_admm[idx], color='blue', linewidth=1.5, label='ADMM Estimate')
    plt.title('Sparsity Pattern (First 100 Features)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sparsity_plot.png')
    plt.show()

if __name__ == "__main__":
    # 1. Setup Data
    A, b, x_true = generate_synthetic_lasso(n=200, p=500, s=20)
    
    # 2. Run Solvers
    lam = 0.5  # Regularization strength
    rho = 1.0  # ADMM penalty
    
    x_admm, hist_admm = admm_lasso(A, b, lam, rho)
    x_gd, hist_gd = subgradient_lasso(A, b, lam)
    
    # 3. Compare Results
    print("\nFinal Objectives:")
    print(f"ADMM: {hist_admm['objval'][-1]:.4f}")
    print(f"GD:   {hist_gd[-1]:.4f}")
    
    plot_results(x_true, x_admm, x_gd, hist_admm, hist_gd)
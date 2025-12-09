# ADMM for Lasso Regression

**Authors:** Donglai Wu, Ori Wang  
**Course:** AM205  
**Date:** December 2025

## Project Overview
This repository contains a Python implementation of the **Alternating Direction Method of Multipliers (ADMM)** algorithm to solve the Lasso regression problem. We compare the efficiency and accuracy of ADMM against a standard **Subgradient Descent** baseline on synthetic high-dimensional data.

The project demonstrates how ADMM effectively decomposes the non-smooth $L_1$ regularization problem into simpler subproblems, achieving faster convergence and better sparsity patterns than gradient-based methods.

## Mathematical Formulation
The Lasso problem is defined as:

$$
\min_{x} \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_1
$$

ADMM solves this by introducing an auxiliary variable $z$ and decoupling the objective:

1.  **x-update (Ridge Regression step):**
    $$(A^\top A + \rho I) x^{k+1} = A^\top b + \rho(z^k - u^k)$$
    
2.  **z-update (Soft Thresholding step):**
    $$z^{k+1} = S_{\lambda/\rho}(x^{k+1} + u^k)$$
    
3.  **u-update (Dual variable):**
    $$u^{k+1} = u^k + x^{k+1} - z^{k+1}$$

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oridmw/am205-final-project-admm.git
   cd am205-final-project-admm
   python lasso_admm.py
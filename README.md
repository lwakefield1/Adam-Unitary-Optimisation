# Adam Optimisation (Adaptive Gradient Method) — Octave Implementation

This repository contains an **Octave/MATLAB implementation of the Adam optimiser** for learning unitary matrices.  
The algorithm builds on the Batch Gradient Descent (BGD) approach but introduces **adaptive moment estimation**, enabling smoother and more stable convergence.

The goal is to approximate a reference unitary matrix `U_ref` by iteratively optimising a parameter matrix `P`,  
which is mapped to a unitary matrix `U_test = UC(P)` using a composite parameterisation.

This method minimises the Frobenius norm–squared difference between `U_ref` and `U_test`,  
and outputs the best unitary matrix found (`U_best`), its associated cost,  
and the iteration number at which convergence occurs.


---

## 🧠 Overview of the Algorithm

The Adam optimiser introduces **moment-based adaptive learning rates**, improving on standard gradient descent by maintaining exponentially weighted averages of both the first moment (mean) and second moment (uncentered variance) of the gradients.

### Algorithm Outline

1. **Reference Unitary Generation**
   - A random complex matrix A is generated and decomposed using QR factorisation:  
     `[Q, R] = qr(A)`  
     ensuring the reference matrix  
     `U_ref = Q * diag(diag(R) ./ abs(diag(R)))`  
     is unitary.

2. **Optimisation Loop (Adam Updates)**
   - Initialise `P ∈ ℝ^(d×d)`
   - Compute `U_test = UC(P)`
   - Evaluate the cost function:  
     `C = ||U_ref - U_test||_F²`
   - Compute the gradient numerically using finite differences:  
     `∂C/∂P_ij ≈ (C(P_ij + ε) - C(P_ij - ε)) / (2ε)`
   - Update first and second moment estimates:  
     `m_t = β₁ m_{t-1} + (1 - β₁) g_t`  
     `v_t = β₂ v_{t-1} + (1 - β₂) g_t²`
   - Bias-correct and update the parameter matrix:  
     `P_t = P_{t-1} - α * (m_t / (1 - β₁ᵗ)) / (sqrt(v_t / (1 - β₂ᵗ)) + ε)`

3. **Convergence and Results**
   - The algorithm terminates if `C < 0.01` or the maximum iteration count is reached.
   - Tracks:
     - Cost per iteration  
     - Best cost achieved  
     - Best unitary matrix (U_best)  
     - Iteration number of the best result

---

## ⚙️ Parameters and Settings

| Parameter | Symbol | Default | Description |
|------------|---------|----------|-------------|
| Learning rate | α | 0.15 | Step size controlling update magnitude |
| First moment decay | β₁ | 0.9 | Controls exponential decay rate of the first moment |
| Second moment decay | β₂ | 0.99 | Controls exponential decay rate of the second moment |
| Numerical stability constant | ε | 1e-8 | Prevents division by zero |
| Finite difference step | Δ | 1e-6 | Perturbation for numerical gradient |
| Iterations | — | 50 | Maximum number of Adam steps |
| Random seed | — | 1111 | Fixed for reproducibility |
| Simulations | — | user-defined | Number of independent optimisation runs |

---

## 📈 Outputs

| Output File | Description |
|--------------|-------------|
| `cost_history_data_adam.csv` | Stores iteration vs. average cost data for all simulations |
| `best_unitary_matrix_adam.mat` | Saves the best unitary matrix \( U_\text{best} \) found |
| Figures | Convergence plots showing cost histories, averages, and percentile ranges |

At the end of execution, the console displays:

Best simulation: #3
Best cost achieved: 2.301e-05
Iteration number (best): 28

Best Unitary Matrix (U_best):
0.7071 + 0.7071i 0.0000 + 0.0000i
0.0000 + 0.0000i 0.7071 + 0.7071i
...

css
Copy code
Saved best unitary matrix to **best_unitary_matrix_adam.mat**

## 🧩 Notes on the Learning Rate (α)

The **learning rate** directly affects the convergence behaviour:
- α = 0.15 provides good convergence for most cases (2 ≤ d ≤ 5).
- Reducing α improves stability for larger matrices but slows convergence.
- Increasing α may accelerate learning but risks divergence.

To change α:
```matlab
learning_rate = 0.15;  % Modify as needed



🔍 Example Run
Enter the value of d (dimensions of matrix): 3
Enter the number of simulations: 20
Data has been saved to cost_history_data_adam.csv.
Best simulation: #4
Best cost achieved: 2.4970e-06
Iteration number (best): 35
Cost after 50 iterations (average simulation): 3.14e-04

📊 Visualisation
The code automatically generates three figures:

Cost History for Each Simulation – individual trajectories with average trend
Percentile Range Analysis – 10th, 50th (median), and 90th percentile cost evolution
Iteration Count Histogram – bar chart showing iterations required for convergence

Ensure you are using the Qt graphics backend in Octave for multiple figure windows:

graphics_toolkit('qt');

This project uses the UC.m function authored by Christoph Spengler (University of Vienna, 2011) for the composite parameterisation of the unitary group U(d).

References

Spengler, C., Huber, M., & Hiesmayr, B. C. (2010).
A composite parameterization of unitary groups, density matrices and subspaces.
Journal of Physics A: Mathematical and Theoretical, 43(38), 385306.
arXiv:1004.5252

Spengler, C., Huber, M., & Hiesmayr, B. C. (2011).
Composite parameterization and Haar measure for all unitary and special unitary groups.
arXiv:1103.3408

© Christoph Spengler (2011), Faculty of Physics, University of Vienna.
All other code © 2025 Liam, released under the MIT License.

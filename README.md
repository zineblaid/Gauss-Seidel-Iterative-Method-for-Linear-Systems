# Gauss-Seidel Iterative Method for Linear Systems

## Overview

This project implements the Gauss-Seidel iterative method in Python to approximate the solution of a linear system:

\[
Ax=b
\]

The program computes the Gauss-Seidel iteration matrix, calculates its eigenvalues and spectral radius, performs iterations until the residual error meets the stopping criterion, and compares the numerical solution with NumPy's built-in solver.

## Features

- Validate that the coefficient matrix is square and nonsingular.
- Compute the Gauss-Seidel iteration matrix.
- Calculate eigenvalues and spectral radius.
- Perform iterative solution updates.
- Track the residual error \(\|Ax-b\|\).
- Stop when the residual error falls below a specified tolerance.
- Display iteration results using Matplotlib.
- Compare the computed solution with `numpy.linalg.solve()`.

## Technologies Used

- Python
- NumPy
- Matplotlib

## Installation

Install the required libraries:

```bash
pip install numpy matplotlib
```

## How to Run

```bash
python main.py
```

## Mathematical Background

The Gauss-Seidel method decomposes the coefficient matrix into:

\[
A=D+E+F
\]

where \(D\) is the diagonal matrix, \(E\) is the strictly lower triangular matrix, and \(F\) is the strictly upper triangular matrix.

The iteration formula is:

\[
x^{(k+1)}=(D+E)^{-1}(b-Fx^{(k)})
\]

The iteration matrix is:

\[
G_{GS}=-(D+E)^{-1}F
\]

A spectral radius smaller than 1 guarantees convergence of the iteration.

## Example

The program solves the system using:

\[
A =
\begin{bmatrix}
1 & 1/2 & 0 \\
1/2 & 1 & 1/2 \\
0 & 1/2 & 1
\end{bmatrix},
\qquad
b =
\begin{bmatrix}
1 \\ 1 \\ 1
\end{bmatrix}
\]

The expected exact solution is approximately:

\[
x =
\begin{bmatrix}
1 \\ 0 \\ 1
\end{bmatrix}
\]

## Project Structure

```text
gauss-seidel-iterative-method/
├── main.py
├── README.md
└── .gitignore
```

## Learning Objectives

- Understand iterative methods for linear systems.
- Implement Gauss-Seidel iterations.
- Analyze convergence using the spectral radius.
- Monitor residual errors.
- Compare iterative solutions with a direct numerical solver.

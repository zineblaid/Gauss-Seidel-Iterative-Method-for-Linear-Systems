
import numpy as np
import matplotlib.pyplot as plt


def gauss_seidel(A, b, x0, epsilon=1e-6, max_iterations=1000):
    """Solve Ax = b using the Gauss-Seidel iterative method."""
    n = A.shape[0]

    D = np.diag(np.diag(A))
    E = np.tril(A, k=-1)
    F = np.triu(A, k=1)

    if np.any(np.diag(A) == 0):
        raise ValueError("The diagonal entries of A must be nonzero.")

    iteration_matrix = -np.linalg.solve(D + E, F)
    eigenvalues = np.linalg.eigvals(iteration_matrix)
    spectral_radius = max(abs(eigenvalues))

    print("\nEigenvalues of the Gauss-Seidel iteration matrix:")
    print(eigenvalues)
    print(f"\nSpectral radius: {spectral_radius:.6f}")

    if spectral_radius >= 1:
        raise ValueError(
            "The spectral radius is >= 1. "
            "Convergence is not guaranteed."
        )

    x = x0.astype(float).copy()
    results = []

    for iteration in range(1, max_iterations + 1):
        x_new = x.copy()

        for i in range(n):
            lower_sum = np.dot(A[i, :i], x_new[:i])
            upper_sum = np.dot(A[i, i + 1:], x[i + 1:])

            x_new[i] = (
                b[i] - lower_sum - upper_sum
            ) / A[i, i]

        error = np.linalg.norm(A @ x_new - b)

        results.append([
            iteration,
            np.array2string(x_new, precision=6),
            f"{error:.6e}"
        ])

        x = x_new

        if error < epsilon:
            print("\nStopping criterion reached.")
            return x, results, spectral_radius

    raise RuntimeError(
        f"Method did not converge within {max_iterations} iterations."
    )


def display_results(results):
    """Display the iteration history in a table."""
    table_data = [["Iteration", "Solution", "Residual Error"]]
    table_data.extend(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
        colWidths=[0.2, 0.5, 0.3]
    )

    plt.title("Gauss-Seidel Method Results", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    A = np.array([
        [1, 1 / 2, 0],
        [1 / 2, 1, 1 / 2],
        [0, 1 / 2, 1]
    ], dtype=float)

    b = np.array([1, 1, 1], dtype=float)
    x0 = np.array([0, 1, 1], dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")

    if b.size != A.shape[0] or x0.size != A.shape[0]:
        raise ValueError("Dimensions of A, b, and x0 are incompatible.")

    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must contain finite values.")

    if np.linalg.matrix_rank(A) < A.shape[0]:
        raise ValueError("Matrix A is singular.")

    x, results, spectral_radius = gauss_seidel(
        A, b, x0, epsilon=1e-6
    )

    display_results(results)

    x_exact = np.linalg.solve(A, b)
    difference = np.linalg.norm(x - x_exact)

    print("\nGauss-Seidel solution:")
    print(x)

    print("\nNumPy solution:")
    print(x_exact)

    print(f"\nDifference between solutions: {difference:.6e}")
    print(f"Number of iterations: {len(results)}")


if __name__ == "__main__":
    main()

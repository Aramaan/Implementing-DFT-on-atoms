import numpy as np

def qr(A):
    """
    Computes the QR decomposition of a matrix using the Gram-Schmidt
    orthonormalization algorithm.
    """
    m, n = A.shape

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

A = np.array([[1,4,8,4],
              [4,2,3,7],
              [8,3,6,9],
              [4,7,9,2]])

print(qr(A))

def qr_E(A, eps=1e-6, max_iterations=1000):
    """
    Computes the eigenvalues and eigenvectors of a matrix using the QR
    algorithm. Assumes that the matrix is real and symmetric.
    """
    n = A.shape[0]
    V = np.eye(n)

    for i in range(max_iterations):
        Q, R = qr(A)

        A = np.dot(R, Q)

        V = np.dot(V, Q)

        off_diag = np.abs(A - np.diag(np.diag(A)))
        if np.max(off_diag) < eps:
            break

    eigenvalues = np.diag(A)
    eigenvectors = V

    return eigenvalues, eigenvectors

("The Eigenvalues are")
print(qr_E(A))
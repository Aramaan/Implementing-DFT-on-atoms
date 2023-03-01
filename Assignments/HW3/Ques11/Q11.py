import numpy as np

def gausselim(A, b):
    """
    Solves a system of linear equations using Gaussian elimination
    with partial pivoting.
    """
    n = A.shape[0]
    A = A.copy()
    b = b.copy()

    P = np.eye(n)

    # Perform the Gaussian elimination with partial pivoting
    for k in range(n-1):
        # Find the row with the largest absolute value in the kth column
        i_max = np.argmax(np.abs(A[k:, k])) + k

        # Swap the rows of A, b, and P
        A[[k, i_max], k:] = A[[i_max, k], k:]
        b[[k, i_max]] = b[[i_max, k]]
        P[[k, i_max], :] = P[[i_max, k], :]

        # Eliminate the subdiagonal elements in the kth column
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k+1:] = A[i, k+1:] - factor * A[k, k+1:]
            b[i] = b[i] - factor * b[k]

    # Back-substitution to solve for x
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    # Apply the permutation matrix to x
    x = np.dot(P, x)

    return x

A = np.array([[4,1,-1,-1],
              [-1,4,0,-1],
              [-1,0,4,-1],
              [-1,-1,-1,4]])

b = np.transpose(np.array([5,0,5,0]))

print(np.transpose(gausselim(A,b)))

A = np.array([[0,1,4,1],
              [3,4,-1,-1],
              [1,-4,1,5],
              [2,-2,1,3]])

b = np.transpose(np.array([-4,3,9,7]))

print(np.transpose(gausselim(A,b)))

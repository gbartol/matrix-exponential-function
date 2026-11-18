"""
EISPACK ORTHES / ORTRAN in Python (0-based indexing).

ORTHES: reduce a real general matrix A to upper Hessenberg form H using
        Householder similarity transforms:   H = Q^T A Q
ORTRAN: accumulate the orthogonal matrix Q from the data ORTHES leaves behind.

low, igh are 0-based inclusive bounds of the active block (0 and n-1 if the
matrix was not balanced).
"""
import numpy as np
from math import sqrt, copysign,ceil


def orthes(A, low=0, igh=None): #Vraća modificiranu matricu A i pomoćni vektor ort koji nam treba kasnije
    A = np.array(A, dtype=float) 
    n = A.shape[0]
    igh = n - 1 if igh is None else igh
    ort = np.zeros(n)

    # Column m-1 is reduced for m = low+1 .. igh-1
    for m in range(low + 1, igh):
        col = m - 1

        # Step 1: scale the column to avoid over/underflow
        scale = np.sum(np.abs(A[m:igh + 1, col]))
        if scale == 0.0:
            continue                      # already zero below subdiagonal

        # Step 2: scaled column and its squared norm
        ort[m:igh + 1] = A[m:igh + 1, col] / scale
        h = np.dot(ort[m:igh + 1], ort[m:igh + 1])

        # Step 3: sign choice avoids cancellation; build Householder vector u
        g = -copysign(sqrt(h), ort[m])
        h = h - ort[m] * g
        ort[m] = ort[m] - g               # u = ort[m:igh+1], P = I - u u^T / h

        # Step 4: apply P from the left to rows m..igh, columns m..n-1
        u = ort[m:igh + 1]
        for j in range(m, n):
            f = np.dot(u, A[m:igh + 1, j]) / h
            A[m:igh + 1, j] -= f * u

        # Step 5: apply P from the right to rows 0..igh, columns m..igh
        for i in range(igh + 1):
            f = np.dot(A[i, m:igh + 1], u) / h
            A[i, m:igh + 1] -= f * u

        # Step 6: store results.  A[m+1:, col] keeps the scaled vector
        # (untouched by steps 4-5), A[m, col] becomes the new subdiagonal.
        ort[m] = scale * ort[m]
        A[m, col] = scale * g

    return A, ort


def ortran(A_packed, ort, low=0, igh=None): #Kreće o identitete i modificira ju pomoću pomoćnog vektora ort, vraća matricu Q
    """Build Q (called Z in EISPACK) from the output of orthes."""
    n = A_packed.shape[0]
    igh = n - 1 if igh is None else igh
    ort = ort.copy()
    Z = np.eye(n)

    # Apply the reflectors in reverse order: Q = P_{low+1} ... P_{igh-1}
    for mp in range(igh - 1, low, -1):
        sub = A_packed[mp, mp - 1]
        if sub == 0.0:
            continue
        # restore the rest of u from the stored column (ort[mp] already set)
        ort[mp + 1:igh + 1] = A_packed[mp + 1:igh + 1, mp - 1]
        for j in range(mp, igh + 1):
            g = np.dot(ort[mp:igh + 1], Z[mp:igh + 1, j])
            # dividing by ort[mp]*sub reproduces the 1/h factor
            g = (g / ort[mp]) / sub
            Z[mp:igh + 1, j] += g * ort[mp:igh + 1]
    return Z

def expm_vec(L, v, terms=25): #Računa skalirani Taylorov red na vektoru
    """e^L v by scaling + Taylor series applied to a vector."""
    s = max(1, ceil(np.linalg.norm(L, 1) / 0.5))
    M = L / s
    for _ in range(s):
        term, acc = v.copy(), v.copy()
        for k in range(1, terms):
            term = M @ term / k
            acc += term
        v = acc
    return v


def expm_lower_hessenberg(L): #Računa i vraća e^H
    n = L.shape[0]
    F = np.zeros((n, n))
    F[:, n - 1] = expm_vec(L, np.eye(n)[:, n - 1])      # last column
    for j in range(n - 1, 0, -1):                        # back-fill the rest
        rhs = L @ F[:, j] - F[:, j:] @ L[j:, j]
        F[:, j - 1] = rhs / L[j - 1, j]                  # needs L[j-1,j] != 0
    return F



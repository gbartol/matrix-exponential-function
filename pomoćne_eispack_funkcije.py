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


def hessenberg(A, low=0, igh=None):
    """Convenience wrapper: returns (H, Q) with A = Q H Q^T."""
    packed, ort = orthes(A, low, igh)
    Q = ortran(packed, ort, low, igh)
    H = np.triu(packed, -1)               # drop the stored Householder data
    return H, Q

EPS = np.finfo(float).eps


# --------------------------------------------------------------------------
# 1. One Francis double-shift QR step on the active window H[l..hi]
# --------------------------------------------------------------------------
def francis_step(H, Z, l, hi, s, t):
    """s = sum of the two shifts, t = their product."""
    n = H.shape[0]
    # first column of (H^2 - s H + t I), only 3 entries are nonzero
    x = H[l, l]**2 + H[l, l+1]*H[l+1, l] - s*H[l, l] + t
    y = H[l+1, l]*(H[l, l] + H[l+1, l+1] - s)
    z = H[l+1, l]*H[l+2, l+1]
    xv = np.array([x, y, z])

    for k in range(l, hi):                 # chase the bulge down the diagonal
        sz = min(3, hi - k + 1)            # reflector size: 3, last one is 2
        if k > l:                          # bulge entries from column k-1
            xv = H[k:k+sz, k-1].copy()
        nv = np.linalg.norm(xv)
        if nv == 0.0:
            continue
        alpha = -copysign(nv, xv[0])       # reflector maps xv -> alpha*e1
        v = xv.copy()
        v[0] -= alpha
        beta = 2.0 / (v @ v)

        if k > l:                          # column k-1 is known analytically
            H[k, k-1] = alpha
            H[k+1:k+sz, k-1] = 0.0
        # left:  rows k..k+sz-1, columns k..n-1
        H[k:k+sz, k:] -= beta*np.outer(v, v @ H[k:k+sz, k:])
        # right: rows 0..min(k+3,hi), columns k..k+sz-1
        r = min(k + 3, hi)
        H[:r+1, k:k+sz] -= beta*np.outer(H[:r+1, k:k+sz] @ v, v)
        # accumulate transformation
        Z[:, k:k+sz] -= beta*np.outer(Z[:, k:k+sz] @ v, v)


# --------------------------------------------------------------------------
# 2. Split a 2x2 block with REAL eigenvalues into two 1x1 blocks
# --------------------------------------------------------------------------
def split_2x2(H, Z, p):
    a, b, c, d = H[p, p], H[p, p+1], H[p+1, p], H[p+1, p+1]
    pp = 0.5*(a - d)
    q = pp*pp + b*c
    if q < 0:                              # complex pair: keep the block
        return
    lam = 0.5*(a + d) + copysign(sqrt(q), pp)       # a real eigenvalue
    x1 = np.array([b, lam - a])            # two ways to write an eigenvector
    x2 = np.array([lam - d, c])
    x = x1 if np.linalg.norm(x1) >= np.linalg.norm(x2) else x2
    x = x/np.linalg.norm(x)
    G = np.array([[x[0], -x[1]], [x[1], x[0]]])     # first column = eigenvector
    H[:, p:p+2] = H[:, p:p+2] @ G
    H[p:p+2, :] = G.T @ H[p:p+2, :]
    Z[:, p:p+2] = Z[:, p:p+2] @ G
    H[p+1, p] = 0.0


# --------------------------------------------------------------------------
# 3. Real Schur form by repeated Francis steps with deflation
# --------------------------------------------------------------------------
def schur_francis(H, Z, maxit=30):
    H, Z = H.copy(), Z.copy()
    n = H.shape[0]
    hi, it = n - 1, 0
    while hi > 0:
        l = hi                             # find start of the active window
        while l > 0:
            s = abs(H[l-1, l-1]) + abs(H[l, l])
            if s == 0.0:
                s = np.abs(H).sum()
            if abs(H[l, l-1]) < EPS*s:
                H[l, l-1] = 0.0
                break
            l -= 1

        if l == hi:                        # 1x1 block converged
            hi -= 1; it = 0
        elif l == hi - 1:                  # 2x2 block converged
            split_2x2(H, Z, l)
            hi -= 2; it = 0
        else:                              # iterate
            it += 1
            if it > maxit*n:
                raise RuntimeError("QR iteration did not converge")
            s = H[hi-1, hi-1] + H[hi, hi]
            t = H[hi-1, hi-1]*H[hi, hi] - H[hi-1, hi]*H[hi, hi-1]
            if it % 10 == 0:               # exceptional shift to break cycles
                sc = abs(H[hi, hi-1]) + abs(H[hi-1, hi-2])
                lam = H[hi, hi]
                s = 2*lam + 1.5*sc
                t = lam*lam + 1.5*sc*lam + sc*sc
            francis_step(H, Z, l, hi, s, t)
    return np.triu(H, -1), Z


# --------------------------------------------------------------------------
# 4. Modified HQR2 back-substitution:  T R = R D
# --------------------------------------------------------------------------
def blocks_of(T):
    n, j, blocks = T.shape[0], 0, []
    while j < n:
        if j < n - 1 and T[j+1, j] != 0.0:
            blocks.append((j, 2)); j += 2
        else:
            blocks.append((j, 1)); j += 1
    return blocks


def solve_sylvester(Tii, Dkk, rhs):
    """Solve Tii X - X Dkk = rhs  (all blocks are at most 2x2)."""
    ni, nk = Tii.shape[0], Dkk.shape[0]
    M = np.kron(np.eye(nk), Tii) - np.kron(Dkk.T, np.eye(ni))
    try:
        x = np.linalg.solve(M, rhs.flatten(order="F"))
    except np.linalg.LinAlgError:          # HQR2-style tiny perturbation
        M = M + EPS*np.abs(Tii).max()*np.eye(M.shape[0])
        x = np.linalg.solve(M, rhs.flatten(order="F"))
    return x.reshape((ni, nk), order="F")


def back_substitution(T):
    n = T.shape[0]
    R, D = np.zeros((n, n)), np.zeros((n, n))
    blocks = blocks_of(T)
    for K, (k0, sk) in enumerate(blocks):
        k1 = k0 + sk
        if sk == 1:
            R[k0, k0] = 1.0
            D[k0, k0] = T[k0, k0]
        else:
            a, b, c, d = T[k0, k0], T[k0, k0+1], T[k0+1, k0], T[k0+1, k0+1]
            pp, m = 0.5*(a - d), 0.5*(a + d)
            w = sqrt(-(pp*pp + b*c))                 # eigenvalues m +/- i w
            R[k0:k1, k0:k1] = [[b, 0.0], [-pp, w]]   # [Re x, Im x]
            D[k0:k1, k0:k1] = [[m, w], [-w, m]]
        Dkk = D[k0:k1, k0:k1]
        for i0, si in reversed(blocks[:K]):          # blocks above, nearest first
            i1 = i0 + si
            rhs = -(T[i0:i1, i1:k1] @ R[i1:k1, k0:k1])
            R[i0:i1, k0:k1] = solve_sylvester(T[i0:i1, i0:i1], Dkk, rhs)
    return R, D


# --------------------------------------------------------------------------
# 5. Driver
# --------------------------------------------------------------------------
def eig_orthes_QRD(A):
    H, Q0 = hessenberg(A)                  # ORTHES + ORTRAN
    T, Q = schur_francis(H, Q0)            # T = Q^T A Q
    R, D = back_substitution(T)            # T R = R D
    return Q, R, D, T

def exp_tD(D, t=1.0):
    n = D.shape[0]
    E = np.zeros((n, n))
    for j, s in blocks_of(D):                 # same 1x1 / 2x2 pattern as D
        if s == 1:
            E[j, j] = np.exp(t*D[j, j])
        else:
            m, w = D[j, j], D[j, j+1]
            c, sn = np.cos(t*w), np.sin(t*w)
            E[j:j+2, j:j+2] = np.exp(t*m)*np.array([[c, sn], [-sn, c]])
    return E

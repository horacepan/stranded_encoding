import numpy as np

PHI = {
    1: 0,
    2: 1,
    3: 3,
    4: 4,
    5: 9
}

def serialize_rows(A, s, delta):
    assert 2 <= s <= 5

    n, d = A.shape
    m = int(np.ceil(n // s))
    enc = np.zeros((m, d))

    scalars = np.array([delta**PHI[i+1] for i in range(s)]).reshape(-1, 1)
    for i in range(m):
        rows = A[i*s: (i+1)*s, :]
        encoded_rows = rows * scalars[:len(rows)]
        enc[i] = encoded_rows.sum(axis=0)
    return enc

def serialize_cols(A, s, delta):
    assert 2 <= s <= 5

    n, d = A.shape
    f = int(np.ceil(d // s))
    enc = np.zeros((n, f))

    scalars = np.array([delta**PHI[i+1] for i in range(s)])
    for i in range(f):
        rows = A[:, i*s: (i+1)*s]
        encoded_rows = rows * scalars[:len(rows)]
        enc[:, i] = encoded_rows.sum(axis=1) # set all rows
    return enc

def pack_vec(X, s, delta):
    r, c = X.shape
    assert c == 1
    scalars = np.array([delta**PHI[i+1] for i in range(s)])
    stranded_X = (X * scalars).sum(axis=1).reshape(-1, 1)
    return stranded_X

def deserialize_vec(enc, s, delta):
    d, _ = enc.shape
    double_shifts = np.array([delta**(2*PHI[i+1]) for i in range(s)])
    results = np.zeros((d, s))
    for i in range(s):
        vals = np.floor_divide(enc, double_shifts[i])
        res = np.mod(vals, delta)
        results[:, i] = res.reshape(-1)

    return results.reshape(-1, 1)

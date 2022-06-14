import pdb
import numpy as np
from stranded_encoding import serialize_rows, serialize_cols, pack_vec, deserialize_vec

def test_serialize_rows():
    A = np.array([
        [1, 2, 3],
        [0, 4, 5]
    ])
    s = 2
    delta = 2**14
    sA = serialize_rows(A, s, delta)
    exp = np.array([
        1, 4*delta + 2, 5*delta + 3
    ])
    assert np.allclose(exp, sA), "Serialized row A doesnt match expected"
    print("Serialize rows test okay!")

def test_serialize_cols():
    A = np.array([
        [1, 2],
        [0, 4],
        [9, 3],
    ])
    s = 2
    delta = 2**14

    exp = np.array([
        [2*delta + 1],
        [4*delta + 0],
        [3*delta + 9],
    ])
    sA = serialize_cols(A, s, delta)
    assert np.allclose(exp, sA)
    print("Serialize cols test okay!")

def test_pack_vec():
    x = np.array([10, 20, 30]).reshape(-1, 1)
    s = 3
    delta = 1000
    exp = np.array([
        10*delta**3 + 10*delta + 10,
        20*delta**3 + 20*delta + 20,
        30*delta**3 + 30*delta + 30,
    ]).reshape(x.shape)
    sx = pack_vec(x, s, delta)
    assert np.allclose(sx, exp), "Packed x with itself doesnt match expected"
    print("Pack vector test okay!")

def test_deserialize():
    s = 3
    delta = 2**16
    v = np.array([
        3*delta**6 + 2*delta**2 + 5,
        4*delta**6 + 1*delta**2 + 0,
        2*delta**6 + 7*delta**2 + 9,
    ]).reshape(-1, 1)
    res = deserialize_vec(v, 3, delta)
    exp = np.array([5, 2, 3, 0, 1, 4, 9, 7, 2]).reshape(-1, 1)
    assert np.allclose(res, exp), "Deserialized vector doesnt match expected"
    print("Deserialize vector test okay!")

def test_deserialize_matvec():
    s = 3
    delta = 2**20
    A = np.random.randint(0, 10, size=(60, 5))
    x = np.random.randint(0, 10, size=(5, 1))
    Ax = A @ x

    sA = serialize_rows(A, s, delta)
    sx = pack_vec(x, s, delta)
    sAx = sA @ sx
    dAx = deserialize_vec(sAx, s, delta)
    assert np.allclose(Ax, dAx), "Deserialized matrix vec product doesnt match expected"
    print("Deserialize matrix vector product okay!")


def run_tests():
    test_serialize_rows()
    test_serialize_cols()
    test_pack_vec()
    test_deserialize()
    test_deserialize_matvec()

if __name__ == '__main__':
    run_tests()

import numpy as np
PHIS = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
}

def encode(As, delta=2**26):
    A1, A2, A3, A4 = As
    n = len(A1)
    E = [0] * n

    for i in range(n):
        E[i] += A1[i]
        E[i] += A2[i] * delta
        E[i] += A3[i] * delta**3
        E[i] += A4[i] * delta**4

    return E

def decode(E, delta=2**26):
    A1 = E % delta
    A2 = (E // delta**2) % delta
    A3 = (E // delta**6) % delta
    A4 = E // delta**8
    return [A1, A2, A3, A4]

def dot(A, B):
    tot = 0
    for i in range(len(A)):
        tot += A[i]*B[i]
    return tot

def main():
    np.random.seed(0)
    n = 4
    size = 3
    delta = 2**26
    As = [np.random.randint(2**8, size=size, dtype=np.uint8).tolist() for _ in range(n)]
    Bs = [np.random.randint(2**8, size=size, dtype=np.uint8).tolist() for _ in range(n)]
    eA = encode(As, delta)
    eB = encode(Bs, delta)
    print("A", As)
    print("B", Bs)
    print(eA)
    print(eB)
    print('----')

    ABs = [dot(a, b) for a, b in zip(As, Bs)]
    eAB = dot(eA, eB)
    print("encoded dot", eAB)
    dABs = decode(dot(eA, eB), delta)
    print(ABs, dABs)
    assert ABs == dABs, "Decoded dot products do not match expected dot products"

if __name__ == '__main__':
    main()

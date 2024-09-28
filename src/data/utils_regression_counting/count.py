# pylint: disable=line-too-long
from pathlib import Path
from subprocess import Popen, PIPE
import numpy as np

#! g++ -O3 count.cpp -o count.out


def C(s: str) -> str:
    with Popen((Path(__file__).parent / "count.out").as_posix(), text=True, stdin=PIPE, stdout=PIPE) as c:
        out, err = c.communicate(s)
    assert not err
    return out


def G(a: np.ndarray) -> str:
    s = map(str, np.ravel(a.astype(int)))
    return f"{len(a)}\n{' '.join(s)}\n"


def process(a):

    def N(n: str) -> np.ndarray: return np.fromstring(n, float, sep=" ")

    # ---------------------------------------------------------------------------- #
    #                                 HOMOMORPHISM                                 #
    # ---------------------------------------------------------------------------- #

    hom = dict()

    A = np.array([[0,1,1,0,0,0],[1,0,1,1,0,0],[1,1,0,0,1,0],[0,1,0,0,1,1],[0,0,1,1,0,1],[0,0,0,1,1,0]])
    hom["boat"] = tuple(map(N, C(f"{G(a)}{G(A)}hom").split("\n")))

    A = np.array([[0,1,1,0,0,0],[1,0,1,1,1,0],[1,1,0,0,1,1],[0,1,0,0,1,0],[0,1,1,1,0,1],[0,0,1,0,1,0]])
    hom["chordal6"] = tuple(map(N, C(f"{G(a)}{G(A)}hom").split("\n")))

    A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    hom["chordal4_1"] = tuple(map(N, C(f"{G(a)}{G(A)}hom").split("\n")))

    A = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,1,0,0]])
    hom["chordal4_4"] = tuple(map(N, C(f"{G(a)}{G(A)}hom").split("\n")))

    A = np.array([[0,1,1,0,0],[1,0,1,1,1],[1,1,0,1,0],[0,1,1,0,1],[0,1,0,1,0]])
    hom["chordal5_13"] = tuple(map(N, C(f"{G(a)}{G(A)}hom").split("\n")))

    A = np.array([[0,1,1,1,1],[1,0,1,0,0],[1,1,0,1,0],[1,0,1,0,1],[1,0,0,1,0]])
    hom["chordal5_31"] = tuple(map(N, C(f"{G(a)}{G(A)}hom").split("\n")))

    A = np.array([[0,1,1,0,1],[1,0,1,0,0],[1,1,0,1,1],[0,0,1,0,1],[1,0,1,1,0]])
    hom["chordal5_24"] = tuple(map(N, C(f"{G(a)}{G(A)}hom").split("\n")))

    # ---------------------------------------------------------------------------- #
    #                                  ISOMORPHISM                                 #
    # ---------------------------------------------------------------------------- #

    iso = dict()

    # ----------------------------------- CYCLE ---------------------------------- #

    cyc = lambda n: np.roll(np.eye(n), 1, axis=0) + np.roll(np.eye(n), 1, axis=1)

    for n in [3, 4, 5, 6, 7, 8]:
        au = n * 2

        g, v, e = C(f"{G(a)}{G(cyc(n))}iso").split("\n")
        iso[f"cycle{n}"] = N(g)/au, N(v)/au, N(e)/au

    # ---------------------------------- CHORDAL --------------------------------- #

    chord = lambda n: (a:=cyc(n), np.put(a, [2 * n - 1, n * n - n + 1], 1), a)[2]

    for n, au in [(4, 4), (5, 2)]:

        g, v, e = C(f"{G(a)}{G(chord(n))}iso").split("\n")
        iso[f"chordal{n}"] = N(g)/au, N(v)/au, N(e)/au

    return hom, iso


if __name__ == "__main__":

    from multiprocessing import Pool
    from tqdm import tqdm

    with Pool(processes=50) as mp:

        data,_ = np.load("graph.npy", allow_pickle=True)
        result = mp.map(func=process, iterable=tqdm(data))

    hom, iso = zip(*result)

    np.save("hom.npy", hom, allow_pickle=True)
    np.save("iso.npy", iso, allow_pickle=True)

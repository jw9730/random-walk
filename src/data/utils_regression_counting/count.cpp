#include <iostream>
#include <string.h>
#include <cassert>

#define MAX_N 40

// graph

static int N;          // num of nodes
static int A[MAX_N][MAX_N];  // adjacent mat

// substr

static int M;          // num of nodes
static int B[MAX_N][MAX_N];  // adjacent mat

// counting

static int G;          // total
static int V[MAX_N];      // node level
static int E[MAX_N][MAX_N];  // edge level

/* -------------------------------------------------------------------------- */
/*                                    COUNT                                   */
/* -------------------------------------------------------------------------- */

// Homomorphism Counting

static int k[MAX_N];
static void hom(int i) {
    if (i == M) {
        G += 1;
        V[k[0]] += 1;
        if (k[0] != k[1])
            E[k[0]][k[1]] += 1;
        return;
    }

    for (int u = 0; u < N; u++) {
        int ok = true;
        for (int j = 0; j < i; j++)
            if (A[u][k[j]] < B[i][j]) {
                ok = false;
                break;
            }

        if (ok) {
            k[i] = u;
            hom(i + 1);
        }
    }
}

// Isomorphism Counting w/
//  duplicated automorphism

static int v[MAX_N];
static void iso(int i) {
    if (i == M) {
        G += 1;
        for (int id_i = 0; id_i < M; V[k[id_i++]] += 1) {
            for (int id_j = 0; id_j < M; id_j++) {
                if (B[id_i][id_j]) E[k[id_i]][k[id_j]] += 1;
            }
        }
        return;
    }

    for (int u = 0; u < N; u++)
        if (v[u] == false) {
            int ok = true;

            for (int j = 0; j < i; j++)
                if (A[u][k[j]] < B[i][j]) {
                    ok = false;
                    break;
                }

            if (ok) {
                k[i] = u;
                v[u] = 1;
                iso(i + 1);
                v[u] = 0;
            }
        }
}

int main(int argc, char *argv[]) {
    std::cin >> N;
    assert(N <= MAX_N);

    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            std::cin >> A[u][v];
        }
    }

    std::cin >> M;
    assert(M <= MAX_N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            std::cin >> B[i][j];
        }
    }

    std::string cmd;
    std::cin >> cmd;

    if (cmd == "hom") hom(0);
    if (cmd == "iso") iso(0);

    std::cout << G << std::endl;

    for (int u = 0; u < N; u++) {
        std::cout << V[u] << " ";
    }

    std::cout << std::endl;

    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            std::cout << E[u][v] << " ";
        }
    }

    return 0;
}

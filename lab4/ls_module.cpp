#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include <limits>

namespace py = pybind11;

static inline int idx2(int i, int j, int n) { return i * n + j; }

// ------------------------------------------------------------
// Objective
// ------------------------------------------------------------
static inline long long objective(const int32_t* dist, int n, const int32_t* costs,
                                  const std::vector<int>& tour) {
    int m = (int)tour.size();
    long long len = 0, cs = 0;
    for (int i = 0; i < m; i++) {
        int a = tour[i];
        int b = tour[(i + 1) % m];
        len += dist[idx2(a, b, n)];
        cs  += costs[a];
    }
    return len + cs;
}

// ------------------------------------------------------------
// Random solution
// ------------------------------------------------------------
static std::vector<int> random_solution(int n, int m, std::mt19937& rng) {
    std::vector<int> nodes(n);
    for (int i = 0; i < n; i++) nodes[i] = i;
    std::shuffle(nodes.begin(), nodes.end(), rng);
    nodes.resize(m);
    std::shuffle(nodes.begin(), nodes.end(), rng);
    return nodes;
}

// ------------------------------------------------------------
// Inter exchange delta/apply
// ------------------------------------------------------------
static inline long long delta_inter_exchange(const int32_t* dist, int n, const int32_t* costs,
                                             const std::vector<int>& tour, int pos, int node_out) {
    int m = (int)tour.size();
    int a = tour[pos];
    int prev = tour[(pos - 1 + m) % m];
    int nxt  = tour[(pos + 1) % m];

    long long old_len = (long long)dist[idx2(prev, a, n)] + dist[idx2(a, nxt, n)];
    long long new_len = (long long)dist[idx2(prev, node_out, n)] + dist[idx2(node_out, nxt, n)];

    long long delta_len  = new_len - old_len;
    long long delta_cost = (long long)costs[node_out] - costs[a];
    return delta_len + delta_cost;
}

static inline void apply_inter_exchange(std::vector<int>& tour, std::vector<uint8_t>& selected,
                                       int pos, int node_out) {
    int a = tour[pos];
    tour[pos] = node_out;
    selected[a] = 0;
    selected[node_out] = 1;
}

// ------------------------------------------------------------
// 2-opt delta/apply
// ------------------------------------------------------------
static inline long long delta_two_opt(const int32_t* dist, int n,
                                      const std::vector<int>& tour, int i, int j) {
    int m = (int)tour.size();
    if (j == i + 1) return 0;
    if (i == 0 && j == m - 1) return 0;

    int a = tour[i];
    int b = tour[i + 1];
    int c = tour[j];
    int d = tour[(j + 1) % m];

    long long old_len = (long long)dist[idx2(a, b, n)] + dist[idx2(c, d, n)];
    long long new_len = (long long)dist[idx2(a, c, n)] + dist[idx2(b, d, n)];
    return new_len - old_len;
}

static inline void apply_two_opt(std::vector<int>& tour, int i, int j) {
    std::reverse(tour.begin() + (i + 1), tour.begin() + (j + 1));
}

// ------------------------------------------------------------
// Candidate lists (directed by score dist(u,v)+cost(v))
// ------------------------------------------------------------
static std::vector<std::vector<int>> build_candidates(const int32_t* dist, int n,
                                                      const int32_t* costs, int k) {
    std::vector<std::vector<int>> cand(n);
    std::vector<std::pair<int,int>> tmp;
    tmp.reserve(n - 1);

    for (int u = 0; u < n; u++) {
        tmp.clear();
        for (int v = 0; v < n; v++) if (v != u) {
            int score = (int)dist[idx2(u, v, n)] + (int)costs[v];
            tmp.push_back({score, v});
        }
        if ((int)tmp.size() > k) {
            std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end(),
                             [](auto &a, auto &b){ return a.first < b.first; });
            tmp.resize(k);
        }
        std::sort(tmp.begin(), tmp.end(),
                  [](auto &a, auto &b){ return a.first < b.first; });

        cand[u].reserve(tmp.size());
        for (auto &p : tmp) cand[u].push_back(p.second);
    }
    return cand;
}

// Build symmetric candidate sets: CandSym[u] = Cand[u] ∪ {v | u in Cand[v]}
static std::vector<std::vector<int>> build_sym_candidates(const std::vector<std::vector<int>>& cand, int n) {
    std::vector<std::vector<int>> sym = cand;
    for (int u = 0; u < n; u++) sym[u].reserve(sym[u].size() * 2);

    for (int v = 0; v < n; v++) {
        for (int u : cand[v]) {
            sym[u].push_back(v);
        }
    }

    for (int u = 0; u < n; u++) {
        std::sort(sym[u].begin(), sym[u].end());
        sym[u].erase(std::unique(sym[u].begin(), sym[u].end()), sym[u].end());
    }
    return sym;
}

static inline bool contains_sorted(const std::vector<int>& vec, int x) {
    return std::binary_search(vec.begin(), vec.end(), x);
}

static inline bool cand_edge_sym(const std::vector<std::vector<int>>& sym, int u, int v) {
    // sym[u] is sorted
    return contains_sorted(sym[u], v);
}

// ------------------------------------------------------------
// Baseline steepest full neighborhood
// ------------------------------------------------------------
static py::tuple steepest_full_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> costs_arr,
    uint32_t seed
) {
    auto dist_buf  = dist_arr.request();
    auto costs_buf = costs_arr.request();

    if (dist_buf.ndim != 2) throw std::runtime_error("dist must be 2D");
    int n = (int)dist_buf.shape[0];
    if ((int)dist_buf.shape[1] != n) throw std::runtime_error("dist must be (n,n)");
    if (costs_buf.ndim != 1 || (int)costs_buf.shape[0] != n) throw std::runtime_error("costs must be (n,)");

    const int32_t* dist  = (const int32_t*)dist_buf.ptr;
    const int32_t* costs = (const int32_t*)costs_buf.ptr;

    int m = (n + 1) / 2;
    std::mt19937 rng(seed);

    std::vector<int> tour = random_solution(n, m, rng);
    std::vector<uint8_t> selected(n, 0);
    for (int v : tour) selected[v] = 1;

    long long evaluated = 0;
    std::vector<int> unselected; unselected.reserve(n);

    while (true) {
        unselected.clear();
        for (int v = 0; v < n; v++) if (!selected[v]) unselected.push_back(v);

        long long best_delta = 0;
        int best_kind = -1; // 0 inter, 1 2opt
        int bi=0,bj=0,bpos=0,bout=0;

        for (int i = 0; i < m; i++) {
            for (int j = i + 2; j < m; j++) {
                if (i == 0 && j == m - 1) continue;
                evaluated++;
                long long d = delta_two_opt(dist, n, tour, i, j);
                if (d < best_delta) {
                    best_delta = d; best_kind = 1; bi=i; bj=j;
                }
            }
        }

        for (int pos = 0; pos < m; pos++) {
            for (int v : unselected) {
                evaluated++;
                long long d = delta_inter_exchange(dist, n, costs, tour, pos, v);
                if (d < best_delta) {
                    best_delta = d; best_kind = 0; bpos=pos; bout=v;
                }
            }
        }

        if (best_kind == -1) break;
        if (best_kind == 0) apply_inter_exchange(tour, selected, bpos, bout);
        else apply_two_opt(tour, bi, bj);
    }

    py::array_t<int32_t> tour_out((py::ssize_t)tour.size());
    auto tbuf = tour_out.request();
    auto* tptr = (int32_t*)tbuf.ptr;
    for (size_t i=0;i<tour.size();i++) tptr[i] = (int32_t)tour[i];

    return py::make_tuple(tour_out, objective(dist, n, costs, tour), evaluated);
}

// ------------------------------------------------------------
// Candidate steepest (FIXED enumeration)
// ------------------------------------------------------------
static py::tuple steepest_candidate_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> costs_arr,
    int k,
    uint32_t seed
) {
    auto dist_buf  = dist_arr.request();
    auto costs_buf = costs_arr.request();

    if (dist_buf.ndim != 2) throw std::runtime_error("dist must be 2D");
    int n = (int)dist_buf.shape[0];
    if ((int)dist_buf.shape[1] != n) throw std::runtime_error("dist must be (n,n)");
    if (costs_buf.ndim != 1 || (int)costs_buf.shape[0] != n) throw std::runtime_error("costs must be (n,)");

    const int32_t* dist  = (const int32_t*)dist_buf.ptr;
    const int32_t* costs = (const int32_t*)costs_buf.ptr;

    int m = (n + 1) / 2;
    std::mt19937 rng(seed);

    std::vector<int> tour = random_solution(n, m, rng);
    std::vector<uint8_t> selected(n, 0);
    for (int v : tour) selected[v] = 1;

    auto cand = build_candidates(dist, n, costs, k);
    auto sym  = build_sym_candidates(cand, n); // undirected union

    long long evaluated = 0;
    std::vector<int> pos(n, -1);

    // helper buffer for unique candidate outs per position
    std::vector<int> outs;
    outs.reserve(3 * k);

    while (true) {
        for (int i = 0; i < m; i++) pos[tour[i]] = i;

        long long best_delta = 0;
        int best_kind = -1; // 0 inter, 1 2opt
        int bpos=0, bout=0, bi=0, bj=0;

        // ---- Intra candidate 2-opt moves ----
        // For each selected u, loop v in sym[u]; if v selected, test the 2 (i,j) variants.
        for (int u = 0; u < n; u++) {
            if (!selected[u]) continue;
            int iu = pos[u];

            for (int v : sym[u]) {
                if (!selected[v]) continue;
                int iv = pos[v];
                if (iu == iv) continue;

                int i = iu, j = iv;
                if (i > j) std::swap(i, j);

                evaluated++;
                long long d1 = delta_two_opt(dist, n, tour, i, j);
                if (d1 < best_delta) {
                    best_delta = d1; best_kind = 1; bi=i; bj=j;
                }

                // second move introducing same candidate edge via predecessors
                if (i > 0 && j > 0) {
                    int p = i - 1;
                    int q = j - 1;
                    if (p > q) std::swap(p, q);

                    evaluated++;
                    long long d2 = delta_two_opt(dist, n, tour, p, q);
                    if (d2 < best_delta) {
                        best_delta = d2; best_kind = 1; bi=p; bj=q;
                    }
                }
            }
        }

        // ---- Inter candidate swaps ----
        // Correct enumeration: for each position i, candidate outs come from CandSym[prev] ∪ CandSym[next] ∪ CandSym[a]
        // If v is unselected, swapping introduces candidate edge by construction.
        for (int i = 0; i < m; i++) {
            int a = tour[i];
            int prev = tour[(i - 1 + m) % m];
            int next = tour[(i + 1) % m];

            outs.clear();
            outs.insert(outs.end(), sym[prev].begin(), sym[prev].end());
            outs.insert(outs.end(), sym[next].begin(), sym[next].end());
            outs.insert(outs.end(), sym[a].begin(), sym[a].end());

            std::sort(outs.begin(), outs.end());
            outs.erase(std::unique(outs.begin(), outs.end()), outs.end());

            for (int v : outs) {
                if (selected[v]) continue;

                // ensure at least one candidate edge is introduced (should be true, but keep strict check)
                if (!cand_edge_sym(sym, prev, v) && !cand_edge_sym(sym, v, next)) continue;

                evaluated++;
                long long d = delta_inter_exchange(dist, n, costs, tour, i, v);
                if (d < best_delta) {
                    best_delta = d; best_kind = 0; bpos=i; bout=v;
                }
            }
        }

        if (best_kind == -1) break;
        if (best_kind == 0) apply_inter_exchange(tour, selected, bpos, bout);
        else apply_two_opt(tour, bi, bj);
    }

    py::array_t<int32_t> tour_out((py::ssize_t)tour.size());
    auto tbuf = tour_out.request();
    auto* tptr = (int32_t*)tbuf.ptr;
    for (size_t i=0;i<tour.size();i++) tptr[i] = (int32_t)tour[i];

    return py::make_tuple(tour_out, objective(dist, n, costs, tour), evaluated);
}

// ------------------------------------------------------------
// Module
// ------------------------------------------------------------
PYBIND11_MODULE(ec_ls, m) {
    m.doc() = "EC Task2: baseline steepest + candidate steepest (2-opt + exchange)";

    m.def("steepest_full", &steepest_full_cpp,
          py::arg("dist"), py::arg("costs"),
          py::arg("seed") = 123);

    m.def("steepest_candidate", &steepest_candidate_cpp,
          py::arg("dist"), py::arg("costs"),
          py::arg("k") = 10,
          py::arg("seed") = 123);
}

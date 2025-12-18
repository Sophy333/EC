#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>

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
// Find directed edge (u->v) position i where tour[i]=u and tour[i+1]=v
// ------------------------------------------------------------
static int find_edge_pos(const std::vector<int>& tour, int u, int v) {
    int m = (int)tour.size();
    for (int i = 0; i < m; i++) {
        if (tour[i] == u && tour[(i + 1) % m] == v) return i;
    }
    return -1;
}

static int find_node_pos(const std::vector<int>& tour, int a) {
    for (int i = 0; i < (int)tour.size(); i++) if (tour[i] == a) return i;
    return -1;
}

// ------------------------------------------------------------
// Deltas + apply
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

static inline void apply_inter_exchange(std::vector<int>& tour,
                                       std::vector<uint8_t>& selected,
                                       int pos, int node_out) {
    int a = tour[pos];
    tour[pos] = node_out;
    selected[a] = 0;
    selected[node_out] = 1;
}

// 2-opt delta for i<j, excluding adjacent
static inline long long delta_two_opt(const int32_t* dist, int n,
                                      const std::vector<int>& tour, int i, int j) {
    int m = (int)tour.size();
    if (j == i + 1) return 0;
    if (i == 0 && j == m - 1) return 0;

    int a = tour[i];
    int b = tour[(i + 1) % m];
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
                if (d < best_delta) { best_delta = d; best_kind = 1; bi=i; bj=j; }
            }
        }

        for (int pos = 0; pos < m; pos++) {
            for (int v : unselected) {
                evaluated++;
                long long d = delta_inter_exchange(dist, n, costs, tour, pos, v);
                if (d < best_delta) { best_delta = d; best_kind = 0; bpos=pos; bout=v; }
            }
        }

        if (best_kind == -1) break;
        if (best_kind == 0) apply_inter_exchange(tour, selected, bpos, bout);
        else apply_two_opt(tour, bi, bj);
    }

    py::array_t<int32_t> out((py::ssize_t)tour.size());
    auto ob = out.request();
    auto* p = (int32_t*)ob.ptr;
    for (size_t i = 0; i < tour.size(); i++) p[i] = (int32_t)tour[i];

    return py::make_tuple(out, objective(dist, n, costs, tour), evaluated);
}

// ------------------------------------------------------------
// LM moves (store defining elements; delta must be recomputed when applicable)
// ------------------------------------------------------------
enum class MoveType : uint8_t { TwoOpt = 0, Swap = 1 };

struct Move {
    MoveType type;

    // TwoOpt stored by removed directed edges (a->b) and (c->d)
    int a,b,c,d;

    // Swap stored by replacing x with v (v must be unselected)
    int x, v;
};

// For 2-opt, lecture statuses:
// 0 remove (edges gone)
// 1 keep but skip now (relative direction mismatch)
// 2 applicable now (same direction OR both reversed)
static inline int twoopt_status_and_positions(const std::vector<int>& tour, const Move& mv,
                                             int& i_out, int& j_out) {
    int m = (int)tour.size();
    int e1f = find_edge_pos(tour, mv.a, mv.b);
    int e1r = find_edge_pos(tour, mv.b, mv.a);
    int e2f = find_edge_pos(tour, mv.c, mv.d);
    int e2r = find_edge_pos(tour, mv.d, mv.c);

    bool e1_exists = (e1f != -1) || (e1r != -1);
    bool e2_exists = (e2f != -1) || (e2r != -1);
    if (!e1_exists || !e2_exists) return 0;

    if (e1f != -1 && e2f != -1) {
        i_out = e1f; j_out = e2f;
        if (i_out > j_out) std::swap(i_out, j_out);
        if (j_out == i_out + 1) return 1;
        if (i_out == 0 && j_out == m - 1) return 1;
        return 2;
    }
    if (e1r != -1 && e2r != -1) {
        i_out = e1r; j_out = e2r;
        if (i_out > j_out) std::swap(i_out, j_out);
        if (j_out == i_out + 1) return 1;
        if (i_out == 0 && j_out == m - 1) return 1;
        return 2;
    }

    // different relative direction -> keep, not applicable now
    return 1;
}

static inline bool swap_applicable_current_pos(const std::vector<int>& tour,
                                               const std::vector<uint8_t>& selected,
                                               const Move& mv,
                                               int& pos_out) {
    if (selected[mv.v]) return false;              // v no longer unselected
    pos_out = find_node_pos(tour, mv.x);
    return (pos_out != -1);
}

// Add new improving moves around a changed position
static void add_local_moves(const int32_t* dist, int n, const int32_t* costs,
                            const std::vector<int>& tour,
                            const std::vector<uint8_t>& selected,
                            int center_pos,
                            std::vector<Move>& LM) {
    int m = (int)tour.size();

    int p0 = (center_pos - 2 + m) % m;
    int p1 = (center_pos - 1 + m) % m;
    int p2 = center_pos;
    int p3 = (center_pos + 1) % m;
    int p4 = (center_pos + 2) % m;
    int pos_list[5] = {p0,p1,p2,p3,p4};

    std::vector<int> unselected;
    unselected.reserve(n);
    for (int v = 0; v < n; v++) if (!selected[v]) unselected.push_back(v);

    // local swaps
    for (int t = 0; t < 5; t++) {
        int pos = pos_list[t];
        int x = tour[pos];
        for (int v : unselected) {
            long long d = delta_inter_exchange(dist, n, costs, tour, pos, v);
            if (d < 0) {
                LM.push_back(Move{MoveType::Swap, 0,0,0,0, x, v});
            }
        }
    }

    // local 2-opt: i in neighborhood, j across the tour
    for (int t = 0; t < 5; t++) {
        int i = pos_list[t];
        for (int j = 0; j < m; j++) {
            if (j <= i + 1) continue;
            if (i == 0 && j == m - 1) continue;
            long long d = delta_two_opt(dist, n, tour, i, j);
            if (d < 0) {
                int a = tour[i];
                int b = tour[(i + 1) % m];
                int c = tour[j];
                int dd = tour[(j + 1) % m];
                LM.push_back(Move{MoveType::TwoOpt, a,b,c,dd, 0,0});
                // also inverted version
                LM.push_back(Move{MoveType::TwoOpt, b,a,dd,c, 0,0});
            }
        }
    }
}

// ------------------------------------------------------------
// Correct LM steepest:
// - Build LM from ALL improving moves initially (including inverted)
// - Each iteration: scan LM and (if applicable now) RECOMPUTE delta
//   - if no longer improving -> remove
//   - choose best improving applicable move and apply
// - After apply: locally add new improving moves around the modified area
// ------------------------------------------------------------
static py::tuple steepest_lm_cpp(
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
    std::vector<Move> LM;
    LM.reserve(200000);

    // initial LM build: all improving 2-opt (plus inverted) + all improving swaps
    std::vector<int> unselected;
    unselected.reserve(n);
    for (int v = 0; v < n; v++) if (!selected[v]) unselected.push_back(v);

    for (int i = 0; i < m; i++) {
        for (int j = i + 2; j < m; j++) {
            if (i == 0 && j == m - 1) continue;
            evaluated++;
            long long d = delta_two_opt(dist, n, tour, i, j);
            if (d < 0) {
                int a = tour[i];
                int b = tour[(i + 1) % m];
                int c = tour[j];
                int dd = tour[(j + 1) % m];
                LM.push_back(Move{MoveType::TwoOpt, a,b,c,dd, 0,0});
                LM.push_back(Move{MoveType::TwoOpt, b,a,dd,c, 0,0});
            }
        }
    }

    for (int pos = 0; pos < m; pos++) {
        int x = tour[pos];
        for (int v : unselected) {
            evaluated++;
            long long d = delta_inter_exchange(dist, n, costs, tour, pos, v);
            if (d < 0) LM.push_back(Move{MoveType::Swap, 0,0,0,0, x, v});
        }
    }

    while (true) {
        long long best_delta = 0;
        int best_idx = -1;
        int best_i=0, best_j=0;    // for 2-opt
        int best_pos=0;            // for swap

        for (int idx = 0; idx < (int)LM.size(); idx++) {
            const Move& mv = LM[idx];

            if (mv.type == MoveType::TwoOpt) {
                int i=0,j=0;
                int st = twoopt_status_and_positions(tour, mv, i, j);
                if (st == 0) { LM[idx] = LM.back(); LM.pop_back(); idx--; continue; }
                if (st == 1) continue;

                // applicable -> recompute delta now
                evaluated++;
                long long d = delta_two_opt(dist, n, tour, i, j);
                if (d >= 0) { LM[idx] = LM.back(); LM.pop_back(); idx--; continue; }

                if (d < best_delta) {
                    best_delta = d;
                    best_idx = idx;
                    best_i = i; best_j = j;
                }
            } else {
                int posx = -1;
                if (!swap_applicable_current_pos(tour, selected, mv, posx)) {
                    // if x disappeared or v became selected -> remove
                    LM[idx] = LM.back(); LM.pop_back(); idx--; continue;
                }

                evaluated++;
                long long d = delta_inter_exchange(dist, n, costs, tour, posx, mv.v);
                if (d >= 0) { LM[idx] = LM.back(); LM.pop_back(); idx--; continue; }

                if (d < best_delta) {
                    best_delta = d;
                    best_idx = idx;
                    best_pos = posx;
                }
            }
        }

        if (best_idx == -1) break;

        Move chosen = LM[best_idx];
        LM[best_idx] = LM.back();
        LM.pop_back();

        if (chosen.type == MoveType::TwoOpt) {
            apply_two_opt(tour, best_i, best_j);
            add_local_moves(dist, n, costs, tour, selected, best_i, LM);
            add_local_moves(dist, n, costs, tour, selected, best_j, LM);
        } else {
            // swap at best_pos
            apply_inter_exchange(tour, selected, best_pos, chosen.v);
            add_local_moves(dist, n, costs, tour, selected, best_pos, LM);
        }
    }

    py::array_t<int32_t> out((py::ssize_t)tour.size());
    auto ob = out.request();
    auto* p = (int32_t*)ob.ptr;
    for (size_t i = 0; i < tour.size(); i++) p[i] = (int32_t)tour[i];

    return py::make_tuple(out, objective(dist, n, costs, tour), evaluated);
}

// ------------------------------------------------------------
// Module
// ------------------------------------------------------------
PYBIND11_MODULE(ec_ls, m) {
    m.doc() = "EC Assignment: baseline steepest + LM improving moves (correct deltas)";

    m.def("steepest_full", &steepest_full_cpp,
          py::arg("dist"), py::arg("costs"), py::arg("seed") = 123);

    m.def("steepest_lm", &steepest_lm_cpp,
          py::arg("dist"), py::arg("costs"), py::arg("seed") = 123);
}

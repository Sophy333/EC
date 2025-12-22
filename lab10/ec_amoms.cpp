#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <stdexcept>

namespace py = pybind11;

// ---------------------------
// Helpers
// ---------------------------
static inline int IDX(int i, int j, int n) { return i * n + j; }

static inline long long obj_cycle(const int32_t* dist, int n, const int32_t* costs,
                                  const std::vector<int>& tour) {
    int m = (int)tour.size();
    long long len = 0, cs = 0;
    for (int i = 0; i < m; i++) {
        int a = tour[i];
        int b = tour[(i + 1) % m];
        len += dist[IDX(a, b, n)];
        cs  += costs[a];
    }
    return len + cs;
}

static inline void build_selected(int n, const std::vector<int>& tour, std::vector<uint8_t>& sel) {
    sel.assign(n, 0);
    for (int v : tour) sel[v] = 1;
}

static inline long long delta_2opt(const int32_t* dist, int n,
                                  const std::vector<int>& tour,
                                  int i, int k) {
    // reverse segment (i+1..k) in cyclic order (we implement on linear indexing, i<k)
    // remove edges (i,i+1) and (k,k+1) add (i,k) and (i+1,k+1)
    int m = (int)tour.size();
    int a = tour[i];
    int b = tour[(i + 1) % m];
    int c = tour[k];
    int d = tour[(k + 1) % m];
    return (long long)dist[IDX(a, c, n)] + dist[IDX(b, d, n)]
         - (long long)dist[IDX(a, b, n)] - dist[IDX(c, d, n)];
}

static inline long long delta_swap_selected_with_unselected(const int32_t* dist, int n, const int32_t* costs,
                                                            const std::vector<int>& tour,
                                                            const std::vector<uint8_t>& sel,
                                                            int pos_i, int v_new) {
    // Replace tour[pos_i]=u with v_new (must be unselected)
    int m = (int)tour.size();
    int u = tour[pos_i];
    int prev = tour[(pos_i - 1 + m) % m];
    int next = tour[(pos_i + 1) % m];

    long long old_edges = (long long)dist[IDX(prev, u, n)] + dist[IDX(u, next, n)];
    long long new_edges = (long long)dist[IDX(prev, v_new, n)] + dist[IDX(v_new, next, n)];
    long long dc = (long long)costs[v_new] - costs[u];

    return (new_edges - old_edges) + dc;
}

// apply 2-opt reversal in-place for i<k (reverse segment i+1..k)
static inline void apply_2opt(std::vector<int>& tour, int i, int k) {
    std::reverse(tour.begin() + (i + 1), tour.begin() + (k + 1));
}

static inline void apply_swap_node(std::vector<int>& tour, int pos_i, int v_new) {
    tour[pos_i] = v_new;
}

// ---------------------------
// Steepest local search (best neighborhood from earlier): 2-opt + inter swap
// ---------------------------
static std::vector<int> steepest_ls(const int32_t* dist, int n, const int32_t* costs,
                                    std::vector<int> tour) {
    int m = (int)tour.size();
    std::vector<uint8_t> sel(n);
    build_selected(n, tour, sel);

    while (true) {
        long long best_delta = 0;
        int best_type = -1; // 0 = 2opt, 1 = inter-swap
        int bi = -1, bk = -1; // for 2opt
        int bpos = -1, bnew = -1; // for inter swap

        // 2-opt neighborhood: i from 0..m-1, k from i+2..m-2 (avoid adjacent and full reverse)
        for (int i = 0; i < m; i++) {
            int i2 = (i + 1) % m;
            for (int k = i + 2; k < m; k++) {
                // avoid breaking adjacency in a way that creates same tour:
                if (i == 0 && k == m - 1) continue;
                long long d = delta_2opt(dist, n, tour, i, k);
                if (d < best_delta) {
                    best_delta = d;
                    best_type = 0;
                    bi = i; bk = k;
                }
            }
        }

        // Inter-route swaps: replace each selected node with each unselected node
        for (int pos = 0; pos < m; pos++) {
            for (int v = 0; v < n; v++) {
                if (sel[v]) continue;
                long long d = delta_swap_selected_with_unselected(dist, n, costs, tour, sel, pos, v);
                if (d < best_delta) {
                    best_delta = d;
                    best_type = 1;
                    bpos = pos; bnew = v;
                }
            }
        }

        if (best_delta >= 0) break;

        if (best_type == 0) {
            apply_2opt(tour, bi, bk);
            // selection unchanged
        } else {
            int old = tour[bpos];
            apply_swap_node(tour, bpos, bnew);
            sel[old] = 0;
            sel[bnew] = 1;
        }
    }

    return tour;
}

// ---------------------------
// Random start (select m nodes and random order)
// ---------------------------
static std::vector<int> random_solution(int n, int m, std::mt19937& rng) {
    std::vector<int> nodes(n);
    for (int i = 0; i < n; i++) nodes[i] = i;
    std::shuffle(nodes.begin(), nodes.end(), rng);
    nodes.resize(m);
    std::shuffle(nodes.begin(), nodes.end(), rng);
    return nodes;
}

// ---------------------------
// Regret insertion helpers (cycle insertion)
// delta inserting v between tour[i] and tour[i+1]:
// dist(a,v)+dist(v,b)-dist(a,b)+cost[v]
// ---------------------------
static inline long long delta_insert_cycle(const int32_t* dist, int n, const int32_t* costs,
                                          const std::vector<int>& tour, int v, int i) {
    int m = (int)tour.size();
    int a = tour[i];
    int b = tour[(i + 1) % m];
    return (long long)dist[IDX(a, v, n)] + dist[IDX(v, b, n)]
         - (long long)dist[IDX(a, b, n)] + costs[v];
}

static inline void best2_insert_for_v(const int32_t* dist, int n, const int32_t* costs,
                                     const std::vector<int>& tour, int v,
                                     long long& best, long long& second, int& best_pos) {
    best = std::numeric_limits<long long>::max();
    second = std::numeric_limits<long long>::max();
    best_pos = -1;
    int m = (int)tour.size();
    for (int i = 0; i < m; i++) {
        long long d = delta_insert_cycle(dist, n, costs, tour, v, i);
        if (d < best) {
            second = best;
            best = d;
            best_pos = i;
        } else if (d < second) {
            second = d;
        }
    }
    if (second == std::numeric_limits<long long>::max()) second = best;
}

static std::vector<int> init_two_node_cycle(const int32_t* dist, int n, const int32_t* costs,
                                            int start, std::vector<uint8_t>& sel) {
    sel.assign(n, 0);
    sel[start] = 1;
    long long best_sc = std::numeric_limits<long long>::max();
    int best_v = -1;
    for (int v = 0; v < n; v++) if (!sel[v]) {
        long long sc = (long long)dist[IDX(start, v, n)] + dist[IDX(v, start, n)] + costs[v];
        if (sc < best_sc) { best_sc = sc; best_v = v; }
    }
    std::vector<int> tour;
    tour.reserve((n + 1) / 2);
    tour.push_back(start);
    tour.push_back(best_v);
    sel[best_v] = 1;
    return tour;
}

static std::vector<int> regret2_construct(const int32_t* dist, int n, const int32_t* costs,
                                          int start, std::mt19937& rng) {
    int m_target = (n + 1) / 2;
    std::vector<uint8_t> sel;
    std::vector<int> tour = init_two_node_cycle(dist, n, costs, start, sel);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    while ((int)tour.size() < m_target) {
        int best_v = -1, best_pos = -1;
        long long best_reg = -1;
        long long best_b1 = 0;

        for (int v = 0; v < n; v++) if (!sel[v]) {
            long long b1, b2; int pos;
            best2_insert_for_v(dist, n, costs, tour, v, b1, b2, pos);
            long long reg = b2 - b1;

            bool choose = false;
            if (best_v == -1) choose = true;
            else if (reg > best_reg) choose = true;
            else if (reg == best_reg && b1 < best_b1) choose = true;
            else if (reg == best_reg && b1 == best_b1 && U(rng) < 0.15) choose = true;

            if (choose) {
                best_v = v; best_pos = pos; best_reg = reg; best_b1 = b1;
            }
        }

        tour.insert(tour.begin() + (best_pos + 1), best_v);
        sel[best_v] = 1;
    }
    return tour;
}

static std::vector<int> weighted_regret2_construct(const int32_t* dist, int n, const int32_t* costs,
                                                   int start, std::mt19937& rng,
                                                   double wR, double wB) {
    int m_target = (n + 1) / 2;
    std::vector<uint8_t> sel;
    std::vector<int> tour = init_two_node_cycle(dist, n, costs, start, sel);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    while ((int)tour.size() < m_target) {
        int best_v = -1, best_pos = -1;
        double best_score = -1e300;

        for (int v = 0; v < n; v++) if (!sel[v]) {
            long long b1, b2; int pos;
            best2_insert_for_v(dist, n, costs, tour, v, b1, b2, pos);
            long long reg = b2 - b1;
            double score = wR * (double)reg + wB * (double)(-b1);

            bool choose = false;
            if (best_v == -1) choose = true;
            else if (score > best_score) choose = true;
            else if (score == best_score && U(rng) < 0.15) choose = true;

            if (choose) {
                best_v = v; best_pos = pos; best_score = score;
            }
        }

        tour.insert(tour.begin() + (best_pos + 1), best_v);
        sel[best_v] = 1;
    }
    return tour;
}

// ---------------------------
// LNS destroy + repair
// Destroy: remove ~30% nodes with probability ∝ badness(node)
// badness = cost + incident edge lengths
// ---------------------------
static std::vector<int> destroy_nodes_badness(const int32_t* dist, int n, const int32_t* costs,
                                              const std::vector<int>& tour,
                                              std::mt19937& rng,
                                              double frac = 0.30) {
    int m = (int)tour.size();
    int k = (int)std::round(frac * m);
    k = std::max(1, std::min(m - 2, k));

    // compute weights
    std::vector<double> w(m, 0.0);
    double sumw = 0.0;
    for (int i = 0; i < m; i++) {
        int v = tour[i];
        int prev = tour[(i - 1 + m) % m];
        int next = tour[(i + 1) % m];
        double bad = (double)costs[v]
                   + (double)dist[IDX(prev, v, n)]
                   + (double)dist[IDX(v, next, n)];
        w[i] = std::max(1e-9, bad);
        sumw += w[i];
    }

    std::vector<uint8_t> removed(m, 0);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    // roulette sampling without replacement (simple O(m*k), m=100 ok)
    for (int t = 0; t < k; t++) {
        double r = U(rng) * sumw;
        int pick = -1;
        double acc = 0.0;
        for (int i = 0; i < m; i++) if (!removed[i]) {
            acc += w[i];
            if (acc >= r) { pick = i; break; }
        }
        if (pick == -1) {
            for (int i = 0; i < m; i++) if (!removed[i]) { pick = i; break; }
        }
        removed[pick] = 1;
        sumw -= w[pick];
        w[pick] = 0.0;
    }

    std::vector<int> partial;
    partial.reserve(m - k);
    for (int i = 0; i < m; i++) if (!removed[i]) partial.push_back(tour[i]);

    // ensure at least 2 nodes
    if ((int)partial.size() < 2) {
        partial.clear();
        partial.push_back(tour[0]);
        partial.push_back(tour[1]);
    }
    return partial;
}

static std::vector<int> repair_by_regret(const int32_t* dist, int n, const int32_t* costs,
                                         std::vector<int> partial,
                                         std::mt19937& rng,
                                         bool weighted = true) {
    int m_target = (n + 1) / 2;
    std::vector<uint8_t> sel;
    build_selected(n, partial, sel);

    std::uniform_real_distribution<double> U(0.0, 1.0);
    // random weights for weighted regret to diversify
    double wR = 1.0, wB = 1.0;
    if (weighted) {
        std::uniform_real_distribution<double> W(0.5, 2.0);
        wR = W(rng);
        wB = W(rng);
    }

    while ((int)partial.size() < m_target) {
        int best_v = -1, best_pos = -1;
        // choose either regret or weighted score
        long long best_reg = -1;
        long long best_b1 = 0;
        double best_score = -1e300;

        for (int v = 0; v < n; v++) if (!sel[v]) {
            long long b1, b2; int pos;
            best2_insert_for_v(dist, n, costs, partial, v, b1, b2, pos);
            long long reg = b2 - b1;

            if (!weighted) {
                bool choose = false;
                if (best_v == -1) choose = true;
                else if (reg > best_reg) choose = true;
                else if (reg == best_reg && b1 < best_b1) choose = true;
                else if (reg == best_reg && b1 == best_b1 && U(rng) < 0.15) choose = true;

                if (choose) { best_v=v; best_pos=pos; best_reg=reg; best_b1=b1; }
            } else {
                double score = wR * (double)reg + wB * (double)(-b1);
                bool choose = false;
                if (best_v == -1) choose = true;
                else if (score > best_score) choose = true;
                else if (score == best_score && U(rng) < 0.15) choose = true;

                if (choose) { best_v=v; best_pos=pos; best_score=score; }
            }
        }

        partial.insert(partial.begin() + (best_pos + 1), best_v);
        sel[best_v] = 1;
    }

    return partial;
}

// ---------------------------
// ILS perturb (k moves)
// - 50%: reverse segment
// - 50%: swap selected with unselected
// ---------------------------
static void perturb(std::vector<int>& tour, const int32_t* dist, int n, const int32_t* costs,
                    std::mt19937& rng, int k_moves) {
    int m = (int)tour.size();
    std::vector<uint8_t> sel;
    build_selected(n, tour, sel);

    std::uniform_int_distribution<int> Upos(0, m - 1);
    std::uniform_int_distribution<int> Unode(0, n - 1);
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int t = 0; t < k_moves; t++) {
        if (U01(rng) < 0.5) {
            int i = Upos(rng), j = Upos(rng);
            if (i > j) std::swap(i, j);
            if (j - i >= 2) {
                std::reverse(tour.begin() + i, tour.begin() + j + 1);
            }
        } else {
            int pos = Upos(rng);
            int u = tour[pos];
            int v = Unode(rng);
            int tries = 0;
            while (sel[v] && tries < 1000) { v = Unode(rng); tries++; }
            if (!sel[v]) {
                tour[pos] = v;
                sel[u] = 0;
                sel[v] = 1;
            }
        }
    }
}

// ---------------------------
// Simple recombination: intersection nodes (order from parent1) + repair
// ---------------------------
static std::vector<int> recombine_intersection_repair(const int32_t* dist, int n, const int32_t* costs,
                                                      const std::vector<int>& p1,
                                                      const std::vector<int>& p2,
                                                      std::mt19937& rng) {
    std::vector<uint8_t> in2(n, 0);
    for (int v : p2) in2[v] = 1;

    std::vector<int> partial;
    partial.reserve((n + 1) / 2);
    for (int v : p1) if (in2[v]) partial.push_back(v);

    // ensure at least 2 nodes
    if ((int)partial.size() < 2) {
        partial.clear();
        partial.push_back(p1[0]);
        partial.push_back(p1[1]);
    }

    // repair with weighted regret
    partial = repair_by_regret(dist, n, costs, partial, rng, true);
    return partial;
}

// ---------------------------
// Elite handling
// ---------------------------
struct Sol {
    std::vector<int> tour;
    long long obj;
};

static void insert_elite(std::vector<Sol>& E, std::unordered_set<long long>& seen_obj,
                         Sol s, int elite_size=20) {
    if (seen_obj.find(s.obj) != seen_obj.end()) return;
    seen_obj.insert(s.obj);
    E.push_back(std::move(s));
    std::sort(E.begin(), E.end(), [](const Sol& a, const Sol& b){ return a.obj < b.obj; });
    if ((int)E.size() > elite_size) {
        // remove worst and also remove its obj from seen
        long long worst = E.back().obj;
        E.pop_back();
        // keep seen_obj simple: don't remove (ok). Or remove for correctness:
        // removing is optional; keeping makes uniqueness stronger.
        (void)worst;
    }
}

static int tournament_pick(const std::vector<Sol>& E, std::mt19937& rng, int tsize=3) {
    std::uniform_int_distribution<int> U(0, (int)E.size() - 1);
    int best = U(rng);
    for (int i = 1; i < tsize; i++) {
        int j = U(rng);
        if (E[j].obj < E[best].obj) best = j;
    }
    return best;
}

// ---------------------------
// AMOMS main
// ---------------------------
static py::tuple amoms_run_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> costs_arr,
    double time_limit_s,
    uint32_t seed,
    bool use_recomb
) {
    auto db = dist_arr.request();
    auto cb = costs_arr.request();
    if (db.ndim != 2) throw std::runtime_error("dist must be 2D");
    int n = (int)db.shape[0];
    if ((int)db.shape[1] != n) throw std::runtime_error("dist must be (n,n)");
    if (cb.ndim != 1 || (int)cb.shape[0] != n) throw std::runtime_error("costs must be (n,)");

    const int32_t* dist = (const int32_t*)db.ptr;
    const int32_t* costs = (const int32_t*)cb.ptr;

    int m = (n + 1) / 2;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // ----- Init elite -----
    std::vector<Sol> E;
    E.reserve(25);
    std::unordered_set<long long> seen_obj;

    for (int i = 0; i < 20; i++) {
        double r = U01(rng);
        std::vector<int> t;

        if (r < 0.40) {
            t = random_solution(n, m, rng);
        } else if (r < 0.70) {
            int start = (int)(rng() % n);
            t = regret2_construct(dist, n, costs, start, rng);
        } else {
            int start = (int)(rng() % n);
            std::uniform_real_distribution<double> W(0.5, 2.0);
            t = weighted_regret2_construct(dist, n, costs, start, rng, W(rng), W(rng));
        }

        t = steepest_ls(dist, n, costs, std::move(t));
        long long fo = obj_cycle(dist, n, costs, t);
        insert_elite(E, seen_obj, Sol{std::move(t), fo}, 20);
    }

    if (E.empty()) {
        auto t = random_solution(n, m, rng);
        t = steepest_ls(dist, n, costs, std::move(t));
        long long fo = obj_cycle(dist, n, costs, t);
        E.push_back(Sol{std::move(t), fo});
    }

    // adaptive operator weights
    double w_lns = 1.0, w_ils = 1.0, w_rec = use_recomb ? 1.0 : 0.0;
    double c_lns = 0.0, c_ils = 0.0, c_rec = 0.0;

    long long best_init = E[0].obj;
    double temp0 = 0.005 * (double)best_init; // 0.5% of objective
    if (temp0 < 1.0) temp0 = 1.0;

    // stats
    long long iters = 0;
    long long acc = 0;
    long long used_lns = 0, used_ils = 0, used_rec = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    auto elapsed_s = [&]() -> double {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d = now - t_start;
        return d.count();
    };

    while (elapsed_s() < time_limit_s) {
        iters++;

        // pick parent
        int ix = tournament_pick(E, rng, 3);
        const Sol& X = E[ix];

        // sample operator
        double sumw = w_lns + w_ils + w_rec;
        double r = U01(rng) * sumw;
        int op = 0; // 0 LNS, 1 ILS, 2 REC
        if (r < w_lns) op = 0;
        else if (r < w_lns + w_ils) op = 1;
        else op = 2;

        std::vector<int> y;

        if (op == 0) {
            used_lns++;
            auto partial = destroy_nodes_badness(dist, n, costs, X.tour, rng, 0.30);
            y = repair_by_regret(dist, n, costs, std::move(partial), rng, true);
        } else if (op == 1) {
            used_ils++;
            y = X.tour;
            std::uniform_int_distribution<int> K(5, 10);
            perturb(y, dist, n, costs, rng, K(rng));
        } else {
            used_rec++;
            int ix2 = tournament_pick(E, rng, 3);
            const Sol& X2 = E[ix2];
            y = recombine_intersection_repair(dist, n, costs, X.tour, X2.tour, rng);
        }

        // intensify
        y = steepest_ls(dist, n, costs, std::move(y));
        long long fy = obj_cycle(dist, n, costs, y);

        // acceptance (SA-ish against parent)
        long long fx = X.obj;
        bool accept = false;

        if (fy < fx) {
            accept = true;
            acc++;
            long long imp = fx - fy;
            if (op == 0) c_lns += (double)imp;
            else if (op == 1) c_ils += (double)imp;
            else c_rec += (double)imp;
        } else {
            double t = elapsed_s() / std::max(1e-9, time_limit_s);
            double temp = temp0 * (1.0 - t);
            if (temp < 1e-9) temp = 1e-9;
            double prob = std::exp(-(double)(fy - fx) / temp);
            if (U01(rng) < prob) {
                accept = true;
                acc++;
            }
        }

        if (accept) {
            insert_elite(E, seen_obj, Sol{std::move(y), fy}, 20);
        }

        if (iters % 50 == 0) {
            w_lns = 0.1 + c_lns;
            w_ils = 0.1 + c_ils;
            w_rec = use_recomb ? (0.1 + c_rec) : 0.0;
            c_lns *= 0.5;
            c_ils *= 0.5;
            c_rec *= 0.5;

            // avoid all-zero
            if (w_lns + w_ils + w_rec < 1e-9) {
                w_lns = 1.0; w_ils = 1.0; w_rec = use_recomb ? 1.0 : 0.0;
            }
        }
    }

    // best is E[0]
    std::sort(E.begin(), E.end(), [](const Sol& a, const Sol& b){ return a.obj < b.obj; });
    const Sol& best = E[0];

    py::array_t<int32_t> best_t((py::ssize_t)best.tour.size());
    auto bt = best_t.request();
    auto* bp = (int32_t*)bt.ptr;
    for (size_t i = 0; i < best.tour.size(); i++) bp[i] = (int32_t)best.tour[i];

    py::dict stats;
    stats["iters"] = iters;
    stats["accepted"] = acc;
    stats["used_lns"] = used_lns;
    stats["used_ils"] = used_ils;
    stats["used_rec"] = used_rec;
    stats["elite_best"] = (long long)best.obj;
    stats["elite_size"] = (int)E.size();

    return py::make_tuple(best_t, (long long)best.obj, stats);
}

// Baseline timing helper: MSLS average time
// One MSLS run = best of `ls_runs` steepest LS runs from random solutions
static py::tuple msls_avg_time_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> costs_arr,
    int runs,
    int ls_runs,
    uint32_t seed
) {
    auto db = dist_arr.request();
    auto cb = costs_arr.request();
    if (db.ndim != 2) throw std::runtime_error("dist must be 2D");
    int n = (int)db.shape[0];
    if ((int)db.shape[1] != n) throw std::runtime_error("dist must be (n,n)");
    if (cb.ndim != 1 || (int)cb.shape[0] != n) throw std::runtime_error("costs must be (n,)");

    const int32_t* dist = (const int32_t*)db.ptr;
    const int32_t* costs = (const int32_t*)cb.ptr;

    int m = (n + 1) / 2;
    std::mt19937 rng(seed);

    std::vector<double> times;
    times.reserve(runs);
    py::array_t<long long> best_objs((py::ssize_t)runs);
    auto bo = best_objs.request();
    auto* bop = (long long*)bo.ptr;

    for (int r = 0; r < runs; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        long long best = std::numeric_limits<long long>::max();

        for (int i = 0; i < ls_runs; i++) {
            auto t = random_solution(n, m, rng);
            t = steepest_ls(dist, n, costs, std::move(t));
            long long f = obj_cycle(dist, n, costs, t);
            if (f < best) best = f;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d = t1 - t0;
        times.push_back(d.count());
        bop[r] = best;
    }

    double avg = 0.0;
    for (double x : times) avg += x;
    avg /= (double)times.size();

    py::array_t<double> times_arr((py::ssize_t)times.size());
    auto ta = times_arr.request();
    auto* tp = (double*)ta.ptr;
    for (size_t i = 0; i < times.size(); i++) tp[i] = times[i];

    return py::make_tuple(avg, times_arr, best_objs);
}

PYBIND11_MODULE(ec_amoms, m) {
    m.doc() = "AMOMS final assignment module (self-contained LS + LNS/ILS/Recomb)";

    m.def("run", &amoms_run_cpp,
          py::arg("dist"), py::arg("costs"),
          py::arg("time_limit_s"),
          py::arg("seed") = 123,
          py::arg("use_recomb") = true);

    m.def("msls_avg_time", &msls_avg_time_cpp,
          py::arg("dist"), py::arg("costs"),
          py::arg("runs") = 20,
          py::arg("ls_runs") = 200,
          py::arg("seed") = 123);
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <chrono>

namespace py = pybind11;

static inline int idx2(int i, int j, int n) { return i * n + j; }

// ============================================================
// Objective
// ============================================================
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

// ============================================================
// Random tour
// ============================================================
static std::vector<int> random_solution(int n, int m, std::mt19937& rng) {
    std::vector<int> nodes(n);
    for (int i = 0; i < n; i++) nodes[i] = i;
    std::shuffle(nodes.begin(), nodes.end(), rng);
    nodes.resize(m);
    std::shuffle(nodes.begin(), nodes.end(), rng);
    return nodes;
}

// ============================================================
// Deltas / moves (steepest: 2-opt + inter exchange)
// ============================================================
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

// ============================================================
// Steepest local search from a tour (C++ internal)
// ============================================================
static void steepest_improve(const int32_t* dist, int n, const int32_t* costs,
                             std::vector<int>& tour) {
    int m = (int)tour.size();
    std::vector<uint8_t> selected(n, 0);
    for (int v : tour) selected[v] = 1;

    std::vector<int> unselected;
    unselected.reserve(n);

    while (true) {
        unselected.clear();
        for (int v = 0; v < n; v++) if (!selected[v]) unselected.push_back(v);

        long long best_delta = 0;
        int best_kind = -1; // 0 inter, 1 2opt
        int bi=0,bj=0,bpos=0,bout=0;

        for (int i = 0; i < m; i++) {
            for (int j = i + 2; j < m; j++) {
                if (i == 0 && j == m - 1) continue;
                long long d = delta_two_opt(dist, n, tour, i, j);
                if (d < best_delta) { best_delta = d; best_kind = 1; bi=i; bj=j; }
            }
        }

        for (int pos = 0; pos < m; pos++) {
            for (int v : unselected) {
                long long d = delta_inter_exchange(dist, n, costs, tour, pos, v);
                if (d < best_delta) { best_delta = d; best_kind = 0; bpos=pos; bout=v; }
            }
        }

        if (best_kind == -1) break;
        if (best_kind == 0) apply_inter_exchange(tour, selected, bpos, bout);
        else apply_two_opt(tour, bi, bj);
    }
}

// ============================================================
// Regret-2 repair (C++ internal)
// ============================================================
static inline long long insertion_cost(const int32_t* dist, int n, const int32_t* costs,
                                       const std::vector<int>& tour, int v, int i) {
    int m = (int)tour.size();
    int a = tour[i];
    int b = tour[(i + 1) % m];
    return (long long)dist[idx2(a, v, n)] + dist[idx2(v, b, n)] - dist[idx2(a, b, n)] + costs[v];
}

static void repair_regret2(const int32_t* dist, int n, const int32_t* costs,
                           std::vector<int>& tour, int m_target, std::mt19937& rng) {
    // ensure unique
    {
        std::vector<int> tmp;
        tmp.reserve(tour.size());
        std::vector<uint8_t> seen(n, 0);
        for (int v : tour) if (!seen[v]) { seen[v] = 1; tmp.push_back(v); }
        tour.swap(tmp);
    }

    std::vector<uint8_t> in_tour(n, 0);
    for (int v : tour) in_tour[v] = 1;

    // ensure at least 2 nodes
    if ((int)tour.size() < 2) {
        std::vector<int> pool;
        pool.reserve(n);
        for (int i = 0; i < n; i++) if (!in_tour[i]) pool.push_back(i);
        std::shuffle(pool.begin(), pool.end(), rng);
        while ((int)tour.size() < 2 && !pool.empty()) {
            int v = pool.back(); pool.pop_back();
            tour.push_back(v);
            in_tour[v] = 1;
        }
    }

    std::uniform_real_distribution<double> ur(0.0, 1.0);

    while ((int)tour.size() < m_target) {
        int best_v = -1;
        long long best_reg = LLONG_MIN;
        long long best_b1 = LLONG_MAX;
        int best_pos = 0;

        for (int v = 0; v < n; v++) {
            if (in_tour[v]) continue;

            long long b1 = LLONG_MAX, b2 = LLONG_MAX;
            int pos1 = 0;

            int m = (int)tour.size();
            for (int i = 0; i < m; i++) {
                long long d = insertion_cost(dist, n, costs, tour, v, i);
                if (d < b1) { b2 = b1; b1 = d; pos1 = i; }
                else if (d < b2) { b2 = d; }
            }
            if (b2 == LLONG_MAX) b2 = b1;
            long long reg = b2 - b1;

            bool choose = false;
            if (best_v == -1) choose = true;
            else if (reg > best_reg) choose = true;
            else if (reg == best_reg && b1 < best_b1) choose = true;
            else if (reg == best_reg && b1 == best_b1 && ur(rng) < 0.25) choose = true;

            if (choose) {
                best_v = v;
                best_reg = reg;
                best_b1 = b1;
                best_pos = pos1;
            }
        }

        tour.insert(tour.begin() + best_pos + 1, best_v);
        in_tour[best_v] = 1;
    }
}

// ============================================================
// Canonical hash for tour uniqueness (rotation + min(forward, reversed))
// We'll hash the canonical sequence with 64-bit rolling hash.
// ============================================================
static uint64_t hash_canonical_tour(const std::vector<int>& tour) {
    int m = (int)tour.size();
    if (m == 0) return 0;

    int minv = tour[0], minpos = 0;
    for (int i = 1; i < m; i++) {
        if (tour[i] < minv) { minv = tour[i]; minpos = i; }
    }

    // build forward rotated
    std::vector<int> fwd(m), rev(m);
    for (int i = 0; i < m; i++) fwd[i] = tour[(minpos + i) % m];

    // reversed tour and rotate to minv
    std::vector<int> t_rev(m);
    for (int i = 0; i < m; i++) t_rev[i] = tour[m - 1 - i];
    int minpos2 = 0;
    for (int i = 0; i < m; i++) if (t_rev[i] == minv) { minpos2 = i; break; }
    for (int i = 0; i < m; i++) rev[i] = t_rev[(minpos2 + i) % m];

    // choose lexicographically smaller
    const std::vector<int>* canon = &fwd;
    if (rev < fwd) canon = &rev;

    // rolling hash
    uint64_t h = 1469598103934665603ull; // FNV offset
    for (int x : *canon) {
        h ^= (uint64_t)(x + 1);
        h *= 1099511628211ull; // FNV prime
    }
    return h;
}

// ============================================================
// Recombination Op1 (common nodes/edges -> subpaths -> random connect)
// We'll use undirected edges.
// ============================================================
struct Edge { int u, v; };
static inline uint64_t edge_key(int u, int v) {
    if (u > v) std::swap(u, v);
    return (uint64_t)u << 32 | (uint32_t)v;
}

static std::vector<std::vector<int>> subpaths_from_common_edges(
    int n, const std::vector<int>& A, const std::vector<int>& B
) {
    int m = (int)A.size();
    std::unordered_set<uint64_t> edgesA;
    edgesA.reserve(m * 2);

    auto add_edges = [&](const std::vector<int>& t, std::unordered_set<uint64_t>& S){
        int mm = (int)t.size();
        for (int i = 0; i < mm; i++) {
            int a = t[i], b = t[(i+1)%mm];
            S.insert(edge_key(a,b));
        }
    };

    add_edges(A, edgesA);
    std::unordered_set<uint64_t> edgesB;
    edgesB.reserve(m * 2);
    add_edges(B, edgesB);

    // common edges
    std::vector<std::vector<int>> adj(n);
    for (auto &e : edgesA) {
        if (edgesB.find(e) != edgesB.end()) {
            int u = (int)(e >> 32);
            int v = (int)(e & 0xffffffffu);
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }

    // common nodes
    std::vector<uint8_t> inA(n,0), inB(n,0);
    for(int x: A) inA[x]=1;
    for(int x: B) inB[x]=1;

    std::vector<uint8_t> visited(n,0);
    std::vector<std::vector<int>> subpaths;

    // endpoints first
    for (int v = 0; v < n; v++) {
        if (!(inA[v] && inB[v])) continue;
        if (adj[v].size() != 1) continue;
        if (visited[v]) continue;

        std::vector<int> path;
        path.push_back(v);
        visited[v] = 1;
        int cur = v, prev = -1;

        while (true) {
            int nxt = -1;
            for (int x : adj[cur]) if (x != prev) { nxt = x; break; }
            if (nxt == -1 || visited[nxt]) break;
            path.push_back(nxt);
            visited[nxt] = 1;
            prev = cur;
            cur = nxt;
        }
        subpaths.push_back(std::move(path));
    }

    // remaining common nodes
    for (int v = 0; v < n; v++) {
        if (!(inA[v] && inB[v])) continue;
        if (visited[v]) continue;

        if (adj[v].empty()) {
            subpaths.push_back({v});
            visited[v] = 1;
            continue;
        }

        std::vector<int> path;
        path.push_back(v);
        visited[v] = 1;
        int cur = v, prev = -1;
        while (true) {
            int nxt = -1;
            for (int x : adj[cur]) if (x != prev) { nxt = x; break; }
            if (nxt == -1 || visited[nxt]) break;
            path.push_back(nxt);
            visited[nxt] = 1;
            prev = cur;
            cur = nxt;
        }
        subpaths.push_back(std::move(path));
    }

    return subpaths;
}

static std::vector<int> recomb_op1(
    int n, int m, const std::vector<int>& A, const std::vector<int>& B, std::mt19937& rng
) {
    auto subpaths = subpaths_from_common_edges(n, A, B);

    std::vector<uint8_t> selected(n,0);
    for (auto &p : subpaths) for (int v : p) selected[v]=1;

    // add random nodes as singletons until reach m
    std::vector<int> candidates;
    candidates.reserve(n);
    for (int v = 0; v < n; v++) if (!selected[v]) candidates.push_back(v);
    std::shuffle(candidates.begin(), candidates.end(), rng);

    while ((int)candidates.size() > 0 && (int)std::count(selected.begin(), selected.end(), 1) < m) {
        int v = candidates.back(); candidates.pop_back();
        subpaths.push_back({v});
        selected[v]=1;
    }

    // shuffle subpaths and randomly flip, concatenate
    std::shuffle(subpaths.begin(), subpaths.end(), rng);
    std::uniform_int_distribution<int> coin(0,1);
    std::vector<int> child;
    child.reserve(m);

    for (auto &p : subpaths) {
        if (coin(rng)) std::reverse(p.begin(), p.end());
        for (int v : p) {
            if ((int)child.size() < m) child.push_back(v);
        }
        if ((int)child.size() >= m) break;
    }

    // fill if short
    if ((int)child.size() < m) {
        std::vector<uint8_t> inChild(n,0);
        for(int v: child) inChild[v]=1;
        std::vector<int> rest;
        for (int v = 0; v < n; v++) if (!inChild[v]) rest.push_back(v);
        std::shuffle(rest.begin(), rest.end(), rng);
        while ((int)child.size() < m) { child.push_back(rest.back()); rest.pop_back(); }
    }

    // ensure unique
    {
        std::vector<uint8_t> seen(n,0);
        std::vector<int> tmp;
        tmp.reserve(m);
        for(int v: child) if(!seen[v]) { seen[v]=1; tmp.push_back(v); }
        child.swap(tmp);
        if ((int)child.size() < m) {
            std::vector<int> rest;
            for (int v = 0; v < n; v++) if (!seen[v]) rest.push_back(v);
            std::shuffle(rest.begin(), rest.end(), rng);
            while ((int)child.size() < m) { child.push_back(rest.back()); rest.pop_back(); }
        }
    }

    return child;
}

// ============================================================
// Recombination Op2 (filter A by nodes in B, then regret repair)
// ============================================================
static std::vector<int> recomb_op2(
    int n, int m, const int32_t* dist, const int32_t* costs,
    const std::vector<int>& A, const std::vector<int>& B, std::mt19937& rng
) {
    std::vector<uint8_t> inB(n,0);
    for (int v : B) inB[v]=1;

    std::vector<int> filtered;
    filtered.reserve(m);
    for (int v : A) if (inB[v]) filtered.push_back(v);

    std::vector<int> child = filtered;
    repair_regret2(dist, n, costs, child, m, rng);
    return child;
}

// ============================================================
// HEA run exposed to Python
// ============================================================
static py::tuple hea_run_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> costs_arr,
    double time_limit_s,
    uint32_t seed,
    int recomb_type,   // 1 = op1, 2 = op2
    double p_ls,       // probability to run LS on offspring
    int tourn_size,
    int pop_size
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
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    // init population (LS always)
    struct Ind { std::vector<int> tour; long long obj; };
    std::vector<Ind> pop;
    pop.reserve(pop_size);

    std::unordered_set<uint64_t> seen;
    seen.reserve(pop_size * 4);

    int attempts = 0;
    while ((int)pop.size() < pop_size && attempts < pop_size * 50) {
        attempts++;
        auto t = random_solution(n, m, rng);
        steepest_improve(dist, n, costs, t);
        long long o = objective(dist, n, costs, t);
        uint64_t hk = hash_canonical_tour(t);
        if (seen.find(hk) != seen.end()) continue;
        seen.insert(hk);
        pop.push_back({std::move(t), o});
    }

    if (pop.empty()) throw std::runtime_error("failed to init population");

    auto best_it = std::min_element(pop.begin(), pop.end(),
                                   [](const Ind& a, const Ind& b){ return a.obj < b.obj; });
    std::vector<int> best_tour = best_it->tour;
    long long best_obj = best_it->obj;

    int iterations = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();
        if (elapsed >= time_limit_s) break;
        iterations++;

        // parents uniform
        std::uniform_int_distribution<int> uid(0, (int)pop.size() - 1);
        int i = uid(rng), j = uid(rng);
        while (j == i) j = uid(rng);

        const auto& A = pop[i].tour;
        const auto& B = pop[j].tour;

        std::vector<int> child;
        if (recomb_type == 2) child = recomb_op2(n, m, dist, costs, A, B, rng);
        else child = recomb_op1(n, m, A, B, rng);

        // LS with probability p_ls
        long long child_obj;
        if (ur(rng) < p_ls) {
            steepest_improve(dist, n, costs, child);
            child_obj = objective(dist, n, costs, child);
        } else {
            child_obj = objective(dist, n, costs, child);
        }

        uint64_t hk = hash_canonical_tour(child);
        if (seen.find(hk) != seen.end()) continue;

        // tournament replacement
        int ts = std::min(tourn_size, (int)pop.size());
        std::vector<int> idxs(pop.size());
        for (int k = 0; k < (int)pop.size(); k++) idxs[k] = k;
        std::shuffle(idxs.begin(), idxs.end(), rng);
        idxs.resize(ts);

        int worst_idx = idxs[0];
        for (int id : idxs) {
            if (pop[id].obj > pop[worst_idx].obj) worst_idx = id;
        }

        if (child_obj < pop[worst_idx].obj) {
            // remove old from seen
            uint64_t oldk = hash_canonical_tour(pop[worst_idx].tour);
            seen.erase(oldk);

            // insert child
            pop[worst_idx] = Ind{std::move(child), child_obj};
            seen.insert(hk);

            if (child_obj < best_obj) {
                best_obj = child_obj;
                best_tour = pop[worst_idx].tour;
            }
        }
    }

    py::array_t<int32_t> out((py::ssize_t)best_tour.size());
    auto ob = out.request();
    auto* p = (int32_t*)ob.ptr;
    for (size_t i = 0; i < best_tour.size(); i++) p[i] = (int32_t)best_tour[i];

    return py::make_tuple(out, (long long)best_obj, iterations);
}

// ============================================================
// Simple exposed steepest functions (same as before)
// ============================================================
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
    auto tour = random_solution(n, m, rng);
    steepest_improve(dist, n, costs, tour);
    long long obj = objective(dist, n, costs, tour);

    py::array_t<int32_t> out((py::ssize_t)tour.size());
    auto ob = out.request();
    auto* p = (int32_t*)ob.ptr;
    for (size_t i = 0; i < tour.size(); i++) p[i] = (int32_t)tour[i];
    return py::make_tuple(out, obj, 0LL);
}

static py::tuple steepest_from_tour_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> costs_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> tour_arr
) {
    auto dist_buf  = dist_arr.request();
    auto costs_buf = costs_arr.request();
    auto tour_buf  = tour_arr.request();

    if (dist_buf.ndim != 2) throw std::runtime_error("dist must be 2D");
    int n = (int)dist_buf.shape[0];
    if ((int)dist_buf.shape[1] != n) throw std::runtime_error("dist must be (n,n)");
    if (costs_buf.ndim != 1 || (int)costs_buf.shape[0] != n) throw std::runtime_error("costs must be (n,)");
    if (tour_buf.ndim != 1) throw std::runtime_error("tour must be 1D");

    const int32_t* dist  = (const int32_t*)dist_buf.ptr;
    const int32_t* costs = (const int32_t*)costs_buf.ptr;
    const int32_t* tptr  = (const int32_t*)tour_buf.ptr;

    int m = (int)tour_buf.shape[0];
    std::vector<int> tour(m);
    for (int i = 0; i < m; i++) tour[i] = (int)tptr[i];

    steepest_improve(dist, n, costs, tour);
    long long obj = objective(dist, n, costs, tour);

    py::array_t<int32_t> out((py::ssize_t)tour.size());
    auto ob = out.request();
    auto* p = (int32_t*)ob.ptr;
    for (size_t i = 0; i < tour.size(); i++) p[i] = (int32_t)tour[i];
    return py::make_tuple(out, obj, 0LL);
}

PYBIND11_MODULE(ec_ls, m) {
    m.doc() = "EC: steepest LS + HEA in C++";

    m.def("steepest_full", &steepest_full_cpp,
          py::arg("dist"), py::arg("costs"), py::arg("seed") = 123);

    m.def("steepest_from_tour", &steepest_from_tour_cpp,
          py::arg("dist"), py::arg("costs"), py::arg("tour"));

    m.def("hea_run", &hea_run_cpp,
          py::arg("dist"),
          py::arg("costs"),
          py::arg("time_limit_s"),
          py::arg("seed") = 123,
          py::arg("recomb_type") = 1,   // 1=op1, 2=op2
          py::arg("p_ls") = 0.7,
          py::arg("tourn_size") = 5,
          py::arg("pop_size") = 20
    );
}

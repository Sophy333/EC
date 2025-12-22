#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include <stdexcept>
#include <limits>

namespace py = pybind11;

static inline int idx2(int i, int j, int n) { return i * n + j; }

// Objective: sum of cycle edges + sum of node costs
static long long objective(const int32_t* dist, int n, const int32_t* costs,
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

// Random solution (select m nodes, random order)
static std::vector<int> random_solution(int n, int m, std::mt19937& rng) {
    std::vector<int> nodes(n);
    for (int i = 0; i < n; i++) nodes[i] = i;
    std::shuffle(nodes.begin(), nodes.end(), rng);
    nodes.resize(m);
    std::shuffle(nodes.begin(), nodes.end(), rng);
    return nodes;
}

// ---- NN end only ----
// delta for adding v at end of OPEN path: dist(last,v) + cost[v]
static std::vector<int> nn_end_only(const int32_t* dist, int n, const int32_t* costs,
                                    int start, int m) {
    std::vector<uint8_t> sel(n, 0);
    std::vector<int> path;
    path.reserve(m);

    path.push_back(start);
    sel[start] = 1;

    while ((int)path.size() < m) {
        int last = path.back();
        long long best = std::numeric_limits<long long>::max();
        int best_v = -1;

        for (int v = 0; v < n; v++) if (!sel[v]) {
            long long d = (long long)dist[idx2(last, v, n)] + costs[v];
            if (d < best) { best = d; best_v = v; }
        }
        path.push_back(best_v);
        sel[best_v] = 1;
    }
    return path; // interpreted as cycle in objective()
}

// ---- NN any position in OPEN path ----
// Positions: begin, end, inside. Choose minimal delta objective.
// begin: dist(v, first) + cost[v]
// end:   dist(last, v) + cost[v]
// inside between a=path[i], b=path[i+1]:
// delta = dist(a,v)+dist(v,b)-dist(a,b) + cost[v]
static std::vector<int> nn_any_position_path(const int32_t* dist, int n, const int32_t* costs,
                                             int start, int m) {
    std::vector<uint8_t> sel(n, 0);
    std::vector<int> path;
    path.reserve(m);

    path.push_back(start);
    sel[start] = 1;

    if (m > 1) {
        // pick second node as best "add at end"
        long long best = std::numeric_limits<long long>::max();
        int best_v = -1;
        for (int v = 0; v < n; v++) if (!sel[v]) {
            long long d = (long long)dist[idx2(start, v, n)] + costs[v];
            if (d < best) { best = d; best_v = v; }
        }
        path.push_back(best_v);
        sel[best_v] = 1;
    }

    while ((int)path.size() < m) {
        long long best_delta = std::numeric_limits<long long>::max();
        int best_v = -1;
        int best_kind = 0; // 0 begin, 1 end, 2 inside
        int best_i = -1;

        int first = path.front();
        int last  = path.back();

        for (int v = 0; v < n; v++) if (!sel[v]) {
            // begin
            {
                long long d = (long long)dist[idx2(v, first, n)] + costs[v];
                if (d < best_delta) { best_delta = d; best_v = v; best_kind = 0; }
            }
            // end
            {
                long long d = (long long)dist[idx2(last, v, n)] + costs[v];
                if (d < best_delta) { best_delta = d; best_v = v; best_kind = 1; }
            }
            // inside
            for (int i = 0; i < (int)path.size() - 1; i++) {
                int a = path[i];
                int b = path[i + 1];
                long long d = (long long)dist[idx2(a, v, n)] + dist[idx2(v, b, n)]
                              - dist[idx2(a, b, n)] + costs[v];
                if (d < best_delta) { best_delta = d; best_v = v; best_kind = 2; best_i = i; }
            }
        }

        if (best_kind == 0) path.insert(path.begin(), best_v);
        else if (best_kind == 1) path.push_back(best_v);
        else path.insert(path.begin() + (best_i + 1), best_v);

        sel[best_v] = 1;
    }

    return path; // interpreted as cycle in objective()
}

// ---- Greedy cycle (cheapest insertion into cycle) ----
// Start with [start, best_second] then cheapest cycle insertion:
// delta = dist(a,v)+dist(v,b)-dist(a,b) + cost[v]
static std::vector<int> greedy_cycle(const int32_t* dist, int n, const int32_t* costs,
                                     int start, int m) {
    std::vector<uint8_t> sel(n, 0);
    std::vector<int> tour;
    tour.reserve(m);

    sel[start] = 1;
    tour.push_back(start);
    if (m == 1) return tour;

    // choose 2nd node minimizing dist(s,v)+dist(v,s)+cost[v]
    long long best_score = std::numeric_limits<long long>::max();
    int best_v = -1;
    for (int v = 0; v < n; v++) if (!sel[v]) {
        long long sc = (long long)dist[idx2(start, v, n)] + dist[idx2(v, start, n)] + costs[v];
        if (sc < best_score) { best_score = sc; best_v = v; }
    }
    tour.push_back(best_v);
    sel[best_v] = 1;

    while ((int)tour.size() < m) {
        long long best_delta = std::numeric_limits<long long>::max();
        int best_ins_v = -1;
        int best_i = -1;

        int mm = (int)tour.size();
        for (int v = 0; v < n; v++) if (!sel[v]) {
            for (int i = 0; i < mm; i++) {
                int a = tour[i];
                int b = tour[(i + 1) % mm];
                long long d = (long long)dist[idx2(a, v, n)] + dist[idx2(v, b, n)]
                              - dist[idx2(a, b, n)] + costs[v];
                if (d < best_delta) { best_delta = d; best_ins_v = v; best_i = i; }
            }
        }

        tour.insert(tour.begin() + (best_i + 1), best_ins_v);
        sel[best_ins_v] = 1;
    }

    return tour;
}

// method: 0 Random, 1 NN_end, 2 NN_any_position_path, 3 Greedy_cycle
static py::tuple construct_run200_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> costs_arr,
    int method,
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

    int reps = 200;
    int actual_reps = reps;

    // greedy methods: "starting from each node" => use nodes 0..min(199,n-1)
    if (method != 0) actual_reps = std::min(reps, n);

    py::array_t<long long> objs((py::ssize_t)actual_reps);
    auto ob = objs.request();
    auto* op = (long long*)ob.ptr;

    long long best_obj = std::numeric_limits<long long>::max();
    std::vector<int> best_tour;

    for (int r = 0; r < actual_reps; r++) {
        std::vector<int> tour;

        if (method == 0) {
            tour = random_solution(n, m, rng);
        } else if (method == 1) {
            tour = nn_end_only(dist, n, costs, r, m);
        } else if (method == 2) {
            tour = nn_any_position_path(dist, n, costs, r, m);
        } else if (method == 3) {
            tour = greedy_cycle(dist, n, costs, r, m);
        } else {
            throw std::runtime_error("Unknown method (0..3)");
        }

        long long obj = objective(dist, n, costs, tour);
        op[r] = obj;

        if (obj < best_obj) {
            best_obj = obj;
            best_tour = std::move(tour);
        }
    }

    py::array_t<int32_t> best((py::ssize_t)best_tour.size());
    auto bb = best.request();
    auto* bp = (int32_t*)bb.ptr;
    for (size_t i = 0; i < best_tour.size(); i++) bp[i] = (int32_t)best_tour[i];

    return py::make_tuple(objs, best, best_obj, actual_reps);
}

PYBIND11_MODULE(ec_construct, m) {
    m.doc() = "Construction heuristics for EC TSP+cost (50% nodes)";
    m.def("run200", &construct_run200_cpp,
          py::arg("dist"), py::arg("costs"), py::arg("method"), py::arg("seed") = 123);
}

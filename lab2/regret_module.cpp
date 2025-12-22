#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

static inline int idx2(int i, int j, int n) { return i * n + j; }

// Objective = cycle edges + sum(costs of selected)
static long long objective_cycle(const int32_t* dist, int n, const int32_t* costs,
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

// delta inserting v between tour[i] and tour[i+1] (cycle)
static inline long long delta_insert_cycle(const int32_t* dist, int n, const int32_t* costs,
                                          const std::vector<int>& tour, int v, int i) {
    int m = (int)tour.size();
    int a = tour[i];
    int b = tour[(i + 1) % m];
    return (long long)dist[idx2(a, v, n)] + dist[idx2(v, b, n)]
         - dist[idx2(a, b, n)] + costs[v];
}

// Find best and second-best insertion delta for v, plus argmin position for best
static inline void best2_insertion_for_v(const int32_t* dist, int n, const int32_t* costs,
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

// Build initial 2-node cycle from start: choose v minimizing dist(s,v)+dist(v,s)+cost[v]
static std::vector<int> init_two_node_cycle(const int32_t* dist, int n, const int32_t* costs,
                                            int start, std::vector<uint8_t>& selected) {
    selected.assign(n, 0);
    selected[start] = 1;

    long long best_score = std::numeric_limits<long long>::max();
    int best_v = -1;
    for (int v = 0; v < n; v++) if (!selected[v]) {
        long long sc = (long long)dist[idx2(start, v, n)] + dist[idx2(v, start, n)] + costs[v];
        if (sc < best_score) { best_score = sc; best_v = v; }
    }

    std::vector<int> tour;
    tour.reserve((n + 1) / 2);
    tour.push_back(start);
    tour.push_back(best_v);
    selected[best_v] = 1;
    return tour;
}

// Method A: pure 2-regret (maximize regret = second-best - best)
static std::vector<int> regret2_cycle(const int32_t* dist, int n, const int32_t* costs,
                                      int start, std::mt19937& rng) {
    int target_m = (n + 1) / 2;
    std::vector<uint8_t> selected;
    std::vector<int> tour = init_two_node_cycle(dist, n, costs, start, selected);

    std::uniform_real_distribution<double> U(0.0, 1.0);

    while ((int)tour.size() < target_m) {
        int best_v = -1;
        int best_pos = -1;
        long long best_bestDelta = 0;
        long long best_regret = -1;

        // tie randomization to avoid determinism
        double tie_p = 0.15;

        for (int v = 0; v < n; v++) if (!selected[v]) {
            long long b1, b2;
            int pos;
            best2_insertion_for_v(dist, n, costs, tour, v, b1, b2, pos);
            long long regret = b2 - b1;

            bool choose = false;
            if (best_v == -1) choose = true;
            else if (regret > best_regret) choose = true;
            else if (regret == best_regret && b1 < best_bestDelta) choose = true;
            else if (regret == best_regret && b1 == best_bestDelta && U(rng) < tie_p) choose = true;

            if (choose) {
                best_v = v;
                best_pos = pos;
                best_bestDelta = b1;
                best_regret = regret;
            }
        }

        tour.insert(tour.begin() + (best_pos + 1), best_v);
        selected[best_v] = 1;
    }

    return tour;
}

// Method B: weighted score = wR * regret + wB * (-bestDelta)
// (we want high regret, and low bestDelta)
static std::vector<int> weighted_regret2_cycle(const int32_t* dist, int n, const int32_t* costs,
                                               int start, std::mt19937& rng,
                                               double wR, double wB) {
    int target_m = (n + 1) / 2;
    std::vector<uint8_t> selected;
    std::vector<int> tour = init_two_node_cycle(dist, n, costs, start, selected);

    std::uniform_real_distribution<double> U(0.0, 1.0);

    while ((int)tour.size() < target_m) {
        int best_v = -1;
        int best_pos = -1;
        double best_score = -1e300;

        // tie randomization
        double tie_p = 0.15;

        for (int v = 0; v < n; v++) if (!selected[v]) {
            long long b1, b2;
            int pos;
            best2_insertion_for_v(dist, n, costs, tour, v, b1, b2, pos);

            long long regret = b2 - b1;
            double score = wR * (double)regret + wB * (double)(-b1);

            bool choose = false;
            if (best_v == -1) choose = true;
            else if (score > best_score) choose = true;
            else if (score == best_score && U(rng) < tie_p) choose = true;

            if (choose) {
                best_v = v;
                best_pos = pos;
                best_score = score;
            }
        }

        tour.insert(tour.begin() + (best_pos + 1), best_v);
        selected[best_v] = 1;
    }

    return tour;
}


// Exposed runner:
// method = 0 -> regret2
// method = 1 -> weighted regret2
// returns: (objs[int64], best_tour[int32], best_obj[int64], reps)
static py::tuple run200_regret_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> costs_arr,
    int method,
    uint32_t seed,
    double wR,
    double wB
) {
    auto dist_buf  = dist_arr.request();
    auto costs_buf = costs_arr.request();

    if (dist_buf.ndim != 2) throw std::runtime_error("dist must be 2D");
    int n = (int)dist_buf.shape[0];
    if ((int)dist_buf.shape[1] != n) throw std::runtime_error("dist must be (n,n)");
    if (costs_buf.ndim != 1 || (int)costs_buf.shape[0] != n) throw std::runtime_error("costs must be (n,)");

    const int32_t* dist  = (const int32_t*)dist_buf.ptr;
    const int32_t* costs = (const int32_t*)costs_buf.ptr;

    int reps = std::min(200, n); // "starting from each node" up to 200
    py::array_t<long long> objs((py::ssize_t)reps);
    auto ob = objs.request();
    auto* op = (long long*)ob.ptr;

    std::mt19937 rng(seed);

    long long best_obj = std::numeric_limits<long long>::max();
    std::vector<int> best_tour;

    for (int start = 0; start < reps; start++) {
        std::vector<int> tour;
        if (method == 0) {
            tour = regret2_cycle(dist, n, costs, start, rng);
        } else if (method == 1) {
            tour = weighted_regret2_cycle(dist, n, costs, start, rng, wR, wB);
        } else {
            throw std::runtime_error("method must be 0 or 1");
        }

        long long obj = objective_cycle(dist, n, costs, tour);
        op[start] = obj;

        if (obj < best_obj) {
            best_obj = obj;
            best_tour = std::move(tour);
        }
    }

    py::array_t<int32_t> best((py::ssize_t)best_tour.size());
    auto bb = best.request();
    auto* bp = (int32_t*)bb.ptr;
    for (size_t i = 0; i < best_tour.size(); i++) bp[i] = (int32_t)best_tour[i];

    return py::make_tuple(objs, best, best_obj, reps);
}

PYBIND11_MODULE(ec_regret, m) {
    m.doc() = "Greedy 2-regret and weighted 2-regret for EC TSP+cost (50% nodes)";
    m.def("run200",
          &run200_regret_cpp,
          py::arg("dist"), py::arg("costs"),
          py::arg("method"),
          py::arg("seed") = 123,
          py::arg("wR") = 1.0,
          py::arg("wB") = 1.0);
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

static inline int idx2(int i, int j, int n) { return i * n + j; }

// -------------------- Objective --------------------
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

// -------------------- Random tour (m nodes) --------------------
static std::vector<int> random_solution(int n, int m, std::mt19937& rng) {
    std::vector<int> nodes(n);
    for (int i = 0; i < n; i++) nodes[i] = i;
    std::shuffle(nodes.begin(), nodes.end(), rng);
    nodes.resize(m);
    std::shuffle(nodes.begin(), nodes.end(), rng);
    return nodes;
}

// -------------------- Deltas + apply --------------------
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

// -------------------- Core steepest LS from a given tour --------------------
static py::tuple steepest_from_vec(const int32_t* dist, int n, const int32_t* costs,
                                   std::vector<int> tour) {
    int m = (int)tour.size();
    if (m <= 1) throw std::runtime_error("tour too small");

    // validate uniqueness + range + build selected
    std::vector<uint8_t> selected(n, 0);
    for (int v : tour) {
        if (v < 0 || v >= n) throw std::runtime_error("tour node out of range");
        if (selected[v]) throw std::runtime_error("tour contains duplicates");
        selected[v] = 1;
    }

    long long evaluated = 0;
    std::vector<int> unselected; unselected.reserve(n);

    while (true) {
        unselected.clear();
        for (int v = 0; v < n; v++) if (!selected[v]) unselected.push_back(v);

        long long best_delta = 0;
        int best_kind = -1; // 0 inter, 1 2opt
        int bi=0,bj=0,bpos=0,bout=0;

        // intra: full 2-opt
        for (int i = 0; i < m; i++) {
            for (int j = i + 2; j < m; j++) {
                if (i == 0 && j == m - 1) continue;
                evaluated++;
                long long d = delta_two_opt(dist, n, tour, i, j);
                if (d < best_delta) { best_delta = d; best_kind = 1; bi=i; bj=j; }
            }
        }

        // inter: swap selected at pos with any unselected
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

// -------------------- Python bindings --------------------
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

    return steepest_from_vec(dist, n, costs, std::move(tour));
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

    return steepest_from_vec(dist, n, costs, std::move(tour));
}

PYBIND11_MODULE(ec_ls, m) {
    m.doc() = "EC Assignment: steepest LS (random start) + steepest LS from given tour";

    m.def("steepest_full", &steepest_full_cpp,
          py::arg("dist"), py::arg("costs"), py::arg("seed") = 123);

    m.def("steepest_from_tour", &steepest_from_tour_cpp,
          py::arg("dist"), py::arg("costs"), py::arg("tour"));
}

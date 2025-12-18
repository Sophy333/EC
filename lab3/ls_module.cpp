#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include <limits>

namespace py = pybind11;

static inline int idx2(int i, int j, int n) { return i*n + j; }

static long long cycle_length(const int32_t* dist, int n, const std::vector<int>& tour) {
    int m = (int)tour.size();
    long long s = 0;
    for (int i = 0; i < m; i++) {
        int a = tour[i];
        int b = tour[(i+1)%m];
        s += dist[idx2(a,b,n)];
    }
    return s;
}

static long long objective(const int32_t* dist, int n, const int32_t* costs, const std::vector<int>& tour) {
    long long len = cycle_length(dist, n, tour);
    long long c = 0;
    for (int v : tour) c += costs[v];
    return len + c;
}

// ---------- Starting solutions ----------

static std::vector<int> random_solution(int n, int m, std::mt19937& rng) {
    std::vector<int> nodes(n);
    for (int i = 0; i < n; i++) nodes[i] = i;
    std::shuffle(nodes.begin(), nodes.end(), rng);
    nodes.resize(m);
    std::shuffle(nodes.begin(), nodes.end(), rng);
    return nodes;
}

static inline long long insertion_delta(const int32_t* dist, int n,
                                        const std::vector<int>& tour,
                                        int node, int pos, int32_t cost_node) {
    // insert node between tour[pos] and tour[pos+1]
    int m = (int)tour.size();
    int a = tour[pos];
    int b = tour[(pos+1)%m];
    long long delta_len = (long long)dist[idx2(a,node,n)] + dist[idx2(node,b,n)] - dist[idx2(a,b,n)];
    return delta_len + cost_node;
}

static inline void best_and_second_insertion(const int32_t* dist, int n, const int32_t* costs,
                                             const std::vector<int>& tour, int node,
                                             long long &best_delta, int &best_pos, long long &second_delta) {
    best_delta = std::numeric_limits<long long>::max();
    second_delta = std::numeric_limits<long long>::max();
    best_pos = 0;
    int m = (int)tour.size();
    int32_t cnode = costs[node];

    for (int pos = 0; pos < m; pos++) {
        long long d = insertion_delta(dist, n, tour, node, pos, cnode);
        if (d < best_delta) {
            second_delta = best_delta;
            best_delta = d;
            best_pos = pos;
        } else if (d < second_delta) {
            second_delta = d;
        }
    }
    if (second_delta == std::numeric_limits<long long>::max()) second_delta = best_delta;
}

// Greedy regret (2-regret) building of cycle size m, start from given start_node
static std::vector<int> greedy_regret_cycle(const int32_t* dist, int n, const int32_t* costs,
                                            int m, int start_node, std::mt19937& rng) {
    std::vector<int> tour;
    tour.reserve(m);
    std::vector<uint8_t> selected(n, 0);

    tour.push_back(start_node);
    selected[start_node] = 1;
    if (m == 1) return tour;

    // pick 2nd node: cheapest "2-cycle" expansion: 2*d(start,v) + cost[v]
    long long best_val = std::numeric_limits<long long>::max();
    int best_node = -1;
    for (int v = 0; v < n; v++) if (!selected[v]) {
        long long d = 2LL*dist[idx2(start_node,v,n)] + costs[v];
        if (d < best_val) { best_val = d; best_node = v; }
    }
    tour.push_back(best_node);
    selected[best_node] = 1;

    while ((int)tour.size() < m) {
        long long best_regret = std::numeric_limits<long long>::min();
        long long best_best_delta = std::numeric_limits<long long>::max();
        int chosen = -1;
        int chosen_pos = 0;

        for (int v = 0; v < n; v++) if (!selected[v]) {
            long long bd, sd; int bp;
            best_and_second_insertion(dist, n, costs, tour, v, bd, bp, sd);
            long long regret = sd - bd;

            // maximize regret; tie-break by smaller best insertion cost
            if (regret > best_regret || (regret == best_regret && bd < best_best_delta)) {
                best_regret = regret;
                best_best_delta = bd;
                chosen = v;
                chosen_pos = bp;
            }
        }

        // insert chosen after chosen_pos
        tour.insert(tour.begin() + (chosen_pos + 1), chosen);
        selected[chosen] = 1;
    }
    return tour;
}

// ---------- Delta moves ----------

static inline long long delta_inter_exchange(const int32_t* dist, int n, const int32_t* costs,
                                             const std::vector<int>& tour,
                                             int pos_i, int node_out) {
    int m = (int)tour.size();
    int a = tour[pos_i];
    int prev = tour[(pos_i-1+m)%m];
    int nxt  = tour[(pos_i+1)%m];

    long long old_len = (long long)dist[idx2(prev,a,n)] + dist[idx2(a,nxt,n)];
    long long new_len = (long long)dist[idx2(prev,node_out,n)] + dist[idx2(node_out,nxt,n)];
    long long delta_len = new_len - old_len;
    long long delta_cost = (long long)costs[node_out] - costs[a];
    return delta_len + delta_cost;
}

static inline void apply_inter_exchange(std::vector<int>& tour, std::vector<uint8_t>& selected,
                                       int pos_i, int node_out) {
    int a = tour[pos_i];
    tour[pos_i] = node_out;
    selected[a] = 0;
    selected[node_out] = 1;
}

static inline long long delta_swap_nodes(const int32_t* dist, int n,
                                         const std::vector<int>& tour, int i, int j) {
    if (i == j) return 0;
    int m = (int)tour.size();
    if (i > j) std::swap(i,j);

    int a = tour[i], b = tour[j];
    int im1 = tour[(i-1+m)%m], ip1 = tour[(i+1)%m];
    int jm1 = tour[(j-1+m)%m], jp1 = tour[(j+1)%m];

    // adjacent forward
    if ((i+1)%m == j) {
        long long old_len = (long long)dist[idx2(im1,a,n)] + dist[idx2(a,b,n)] + dist[idx2(b,jp1,n)];
        long long new_len = (long long)dist[idx2(im1,b,n)] + dist[idx2(b,a,n)] + dist[idx2(a,jp1,n)];
        return new_len - old_len;
    }
    // adjacent cyclic (0 and m-1)
    if (i == 0 && j == m-1) {
        long long old_len = (long long)dist[idx2(jm1,b,n)] + dist[idx2(b,a,n)] + dist[idx2(a,ip1,n)];
        long long new_len = (long long)dist[idx2(jm1,a,n)] + dist[idx2(a,b,n)] + dist[idx2(b,ip1,n)];
        return new_len - old_len;
    }

    long long old_len = (long long)dist[idx2(im1,a,n)] + dist[idx2(a,ip1,n)]
                      + (long long)dist[idx2(jm1,b,n)] + dist[idx2(b,jp1,n)];
    long long new_len = (long long)dist[idx2(im1,b,n)] + dist[idx2(b,ip1,n)]
                      + (long long)dist[idx2(jm1,a,n)] + dist[idx2(a,jp1,n)];
    return new_len - old_len;
}

static inline void apply_swap_nodes(std::vector<int>& tour, int i, int j) {
    std::swap(tour[i], tour[j]);
}

static inline long long delta_two_opt(const int32_t* dist, int n,
                                      const std::vector<int>& tour, int i, int j) {
    int m = (int)tour.size();
    // disallow adjacent edges
    if ((i+1)%m == j || (j+1)%m == i) return 0;

    int a = tour[i];
    int b = tour[(i+1)%m];
    int c = tour[j];
    int d = tour[(j+1)%m];

    long long old_len = (long long)dist[idx2(a,b,n)] + dist[idx2(c,d,n)];
    long long new_len = (long long)dist[idx2(a,c,n)] + dist[idx2(b,d,n)];
    return new_len - old_len;
}

static inline void apply_two_opt(std::vector<int>& tour, int i, int j) {
    // reverse segment (i+1 .. j)
    std::reverse(tour.begin() + (i+1), tour.begin() + (j+1));
}

// ---------- Local Search ----------
enum SearchType { STEEPEST = 0, GREEDY = 1 };
enum IntraType  { SWAP = 0, TWO_OPT = 1 };
enum StartType  { START_RANDOM = 0, START_GREEDY_REGRET = 1 };

static py::tuple local_search_cpp(py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_arr,
                                  py::array_t<int32_t, py::array::c_style | py::array::forcecast> costs_arr,
                                  int m,
                                  int search_type,
                                  int intra_type,
                                  int start_type,
                                  uint32_t seed,
                                  int start_node) {
    auto dist_buf = dist_arr.request();
    auto costs_buf = costs_arr.request();

    if (dist_buf.ndim != 2) throw std::runtime_error("dist must be 2D");
    int n = (int)dist_buf.shape[0];
    if ((int)dist_buf.shape[1] != n) throw std::runtime_error("dist must be (n,n)");
    if (costs_buf.ndim != 1 || (int)costs_buf.shape[0] != n) throw std::runtime_error("costs must be (n,)");

    const int32_t* dist = (const int32_t*)dist_buf.ptr;
    const int32_t* costs = (const int32_t*)costs_buf.ptr;

    std::mt19937 rng(seed);

    // start solution
    std::vector<int> tour;
    if (start_type == START_RANDOM) {
        tour = random_solution(n, m, rng);
    } else {
        if (start_node < 0) {
            std::uniform_int_distribution<int> U(0, n-1);
            start_node = U(rng);
        }
        tour = greedy_regret_cycle(dist, n, costs, m, start_node, rng);
    }

    std::vector<uint8_t> selected(n, 0);
    for (int v : tour) selected[v] = 1;

    long long cur_obj = objective(dist, n, costs, tour);
    long long evaluated = 0;

    // For greedy randomization:
    // We randomize which kind to try first each iteration,
    // and within each kind we randomize index order (no full evaluation then shuffle).
    std::vector<int> pos_perm(m);
    for (int i = 0; i < m; i++) pos_perm[i] = i;

    std::vector<int> unselected_nodes; unselected_nodes.reserve(n);

    auto build_unselected = [&](){
        unselected_nodes.clear();
        for (int v = 0; v < n; v++) if (!selected[v]) unselected_nodes.push_back(v);
    };

    while (true) {
        bool improved = false;

        if (search_type == STEEPEST) {
            long long best_delta = 0;
            int best_kind = -1; // 0 inter, 1 swap, 2 2opt
            int bi=0, bj=0, bpos=0, bout=0;

            build_unselected();

            // intra scan
            if (intra_type == SWAP) {
                for (int i = 0; i < m; i++) for (int j = i+1; j < m; j++) {
                    evaluated++;
                    long long d = delta_swap_nodes(dist, n, tour, i, j);
                    if (d < best_delta) {
                        best_delta = d; best_kind = 1; bi=i; bj=j;
                    }
                }
            } else {
                for (int i = 0; i < m; i++) for (int j = i+2; j < m; j++) {
                    if (i==0 && j==m-1) continue;
                    evaluated++;
                    long long d = delta_two_opt(dist, n, tour, i, j);
                    if (d < best_delta) {
                        best_delta = d; best_kind = 2; bi=i; bj=j;
                    }
                }
            }

            // inter scan
            for (int pos = 0; pos < m; pos++) {
                for (int v : unselected_nodes) {
                    evaluated++;
                    long long d = delta_inter_exchange(dist, n, costs, tour, pos, v);
                    if (d < best_delta) {
                        best_delta = d; best_kind = 0; bpos=pos; bout=v;
                    }
                }
            }

            if (best_kind == -1) break; // no improving
            if (best_kind == 0) apply_inter_exchange(tour, selected, bpos, bout);
            else if (best_kind == 1) apply_swap_nodes(tour, bi, bj);
            else apply_two_opt(tour, bi, bj);

            cur_obj += best_delta;
            improved = true;

        } else { // GREEDY
            // pick which neighborhood kind first (0=intra, 1=inter), random each iteration
            std::uniform_int_distribution<int> coin(0,1);
            int first = coin(rng);

            auto try_intra = [&]() -> bool {
                if (intra_type == SWAP) {
                    // randomized i order, randomized j order per i
                    std::vector<int> perm(m);
                    for (int i=0;i<m;i++) perm[i]=i;
                    std::shuffle(perm.begin(), perm.end(), rng);

                    for (int ii=0; ii<m; ii++) {
                        int i = perm[ii];
                        std::vector<int> js;
                        js.reserve(m-1);
                        for (int j=0;j<m;j++) if (j!=i) js.push_back(j);
                        std::shuffle(js.begin(), js.end(), rng);

                        for (int j : js) {
                            if (i==j) continue;
                            evaluated++;
                            long long d = delta_swap_nodes(dist, n, tour, i, j);
                            if (d < 0) {
                                apply_swap_nodes(tour, i, j);
                                cur_obj += d;
                                return true;
                            }
                        }
                    }
                } else { // TWO_OPT
                    std::vector<int> perm(m);
                    for (int i=0;i<m;i++) perm[i]=i;
                    std::shuffle(perm.begin(), perm.end(), rng);

                    for (int ii=0; ii<m; ii++) {
                        int i = perm[ii];
                        std::vector<int> js;
                        js.reserve(m);
                        for (int j=0;j<m;j++) js.push_back(j);
                        std::shuffle(js.begin(), js.end(), rng);

                        for (int j : js) {
                            if (j <= i+1) continue;
                            if (i==0 && j==m-1) continue;
                            evaluated++;
                            long long d = delta_two_opt(dist, n, tour, i, j);
                            if (d < 0) {
                                apply_two_opt(tour, i, j);
                                cur_obj += d;
                                return true;
                            }
                        }
                    }
                }
                return false;
            };

            auto try_inter = [&]() -> bool {
                build_unselected();
                // randomize positions and unselected nodes
                for (int i = 0; i < m; i++) pos_perm[i] = i;
                std::shuffle(pos_perm.begin(), pos_perm.end(), rng);
                std::shuffle(unselected_nodes.begin(), unselected_nodes.end(), rng);

                for (int pos : pos_perm) {
                    for (int v : unselected_nodes) {
                        evaluated++;
                        long long d = delta_inter_exchange(dist, n, costs, tour, pos, v);
                        if (d < 0) {
                            apply_inter_exchange(tour, selected, pos, v);
                            cur_obj += d;
                            return true;
                        }
                    }
                }
                return false;
            };

            if (first == 0) {
                if (try_intra()) improved = true;
                else if (try_inter()) improved = true;
            } else {
                if (try_inter()) improved = true;
                else if (try_intra()) improved = true;
            }

            if (!improved) break;
        }
    }

    // return tour + obj + evaluated
    py::array_t<int32_t> tour_out((py::ssize_t)tour.size());
    auto tbuf = tour_out.request();
    int32_t* tptr = (int32_t*)tbuf.ptr;
    for (size_t i=0;i<tour.size();i++) tptr[i] = (int32_t)tour[i];

    long long final_obj = objective(dist, n, costs, tour);
    return py::make_tuple(tour_out, (long long)final_obj, (long long)evaluated);
}

PYBIND11_MODULE(ec_ls, m) {
    m.doc() = "Evolutionary Computing local search for TSP-like subset cycle";
    m.def("local_search", &local_search_cpp,
          py::arg("dist"), py::arg("costs"), py::arg("m"),
          py::arg("search_type"), py::arg("intra_type"), py::arg("start_type"),
          py::arg("seed"), py::arg("start_node") = -1);
}

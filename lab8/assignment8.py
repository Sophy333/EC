import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ec_ls

def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))

def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    dist = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(int(xi) - int(xj), int(yi) - int(yj))
            dist[i, j] = round_half_up(d)
            dist[j, i] = dist[i, j]
    return dist

def read_instance_csv(path: str, sep=";"):
    df = pd.read_csv(path, sep=sep, header=None)
    coords = df[[0, 1]].to_numpy(dtype=np.int32)
    costs  = df[2].to_numpy(dtype=np.int32)
    dist = build_distance_matrix(coords)
    return coords, dist, costs


def node_mask_int(tour, n):
    mask = 0
    for v in tour:
        mask |= (1 << int(v))
    return mask

def edge_id_undirected(u, v, n):
    if u > v:
        u, v = v, u
    return (u * (2*n - u - 1)) // 2 + (v - u - 1)

def edge_mask_int(tour, n):
    m = len(tour)
    mask = 0
    for i in range(m):
        a = int(tour[i])
        b = int(tour[(i + 1) % m])
        eid = edge_id_undirected(a, b, n)
        mask |= (1 << eid)
    return mask

def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def generate_greedy_optima(dist, costs, num=1000, base_seed=123):
    if not hasattr(ec_ls, "greedy_full"):
        raise AttributeError("ec_ls.greedy_full not found. Rebuild C++ module with greedy_full().")

    n = len(costs)
    tours = []
    objs = np.zeros(num, dtype=np.int64)

    for k in range(num):
        tour, obj, _ = ec_ls.greedy_full(dist, costs, int(base_seed + k))
        tour = np.asarray(tour, dtype=int)
        tours.append(tour)
        objs[k] = int(obj)

    return tours, objs

def very_good_solution_msls(dist, costs, iters=200, base_seed=90000):
    best_obj = None
    best_tour = None
    for i in range(iters):
        tour, obj, _ = ec_ls.steepest_full(dist, costs, int(base_seed + i))
        obj = int(obj)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_tour = np.asarray(tour, dtype=int)
    return best_tour, int(best_obj)


def similarity_to_reference(masks, ref_mask, exclude_index=None):
    s = np.zeros(len(masks), dtype=float)
    for i, mk in enumerate(masks):
        s[i] = (mk & ref_mask).bit_count()
    if exclude_index is not None:
        s[exclude_index] = np.nan
    return s

def avg_similarity_all(masks):
    N = len(masks)
    sums = np.zeros(N, dtype=np.int64)
    for i in range(N):
        mi = masks[i]
        for j in range(i + 1, N):
            c = (mi & masks[j]).bit_count()
            sums[i] += c
            sums[j] += c
    return sums / (N - 1)

def scatter_plot(x, y, title, xlabel="Objective", ylabel="Similarity"):
    mask = ~np.isnan(y)
    x2 = np.asarray(x)[mask]
    y2 = np.asarray(y)[mask]
    r = pearson_corr(x2, y2)

    plt.figure(figsize=(7, 5))
    plt.scatter(x2, y2, s=10, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nPearson r = {r:.4f}")
    plt.tight_layout()
    plt.show()

    return r

def analyze_instance(name, dist, costs, num_optima=1000, base_seed=123):
    n = len(costs)

    tours, objs = generate_greedy_optima(dist, costs, num=num_optima, base_seed=base_seed)

    node_masks = [node_mask_int(t, n) for t in tours]
    edge_masks = [edge_mask_int(t, n) for t in tours]

    best_idx = int(np.argmin(objs))
    best_tour = tours[best_idx]
    best_obj = int(objs[best_idx])
    best_node_ref = node_masks[best_idx]
    best_edge_ref = edge_masks[best_idx]

    vg_tour, vg_obj = very_good_solution_msls(dist, costs, iters=200, base_seed=99999)
    vg_node_ref = node_mask_int(vg_tour, n)
    vg_edge_ref = edge_mask_int(vg_tour, n)

    print(f"\n[{name}] Generated {num_optima} greedy local optima.")
    print(f"[{name}] Best of 1000: obj={best_obj}")
    print(f"[{name}] Very good (MSLS-steepest best of 200): obj={vg_obj}")

    results = []

    avg_nodes = avg_similarity_all(node_masks)
    avg_edges = avg_similarity_all(edge_masks)

    r1 = scatter_plot(objs, avg_nodes, f"{name} | Avg similarity to all optima | Common selected nodes",
                      ylabel="Avg # common selected nodes")
    r2 = scatter_plot(objs, avg_edges, f"{name} | Avg similarity to all optima | Common edges",
                      ylabel="Avg # common edges")

    results.append((name, "avg_to_all", "nodes", r1))
    results.append((name, "avg_to_all", "edges", r2))

    sim_best_nodes = similarity_to_reference(node_masks, best_node_ref, exclude_index=best_idx)
    sim_best_edges = similarity_to_reference(edge_masks, best_edge_ref, exclude_index=best_idx)

    r3 = scatter_plot(objs, sim_best_nodes, f"{name} | Similarity to best of 1000 | Common selected nodes",
                      ylabel="# common selected nodes")
    r4 = scatter_plot(objs, sim_best_edges, f"{name} | Similarity to best of 1000 | Common edges",
                      ylabel="# common edges")

    results.append((name, "to_best1000", "nodes", r3))
    results.append((name, "to_best1000", "edges", r4))

    sim_vg_nodes = similarity_to_reference(node_masks, vg_node_ref, exclude_index=None)
    sim_vg_edges = similarity_to_reference(edge_masks, vg_edge_ref, exclude_index=None)

    r5 = scatter_plot(objs, sim_vg_nodes, f"{name} | Similarity to very good solution | Common selected nodes",
                      ylabel="# common selected nodes")
    r6 = scatter_plot(objs, sim_vg_edges, f"{name} | Similarity to very good solution | Common edges",
                      ylabel="# common edges")

    results.append((name, "to_very_good", "nodes", r5))
    results.append((name, "to_very_good", "edges", r6))

    return {
        "name": name,
        "correlations": results,
        "best_of_1000": (best_obj, best_tour.tolist()),
        "very_good": (vg_obj, vg_tour.tolist()),
    }


def main():
    _, distA, costsA = read_instance_csv("TSPA.csv")
    _, distB, costsB = read_instance_csv("TSPB.csv")

    outA = analyze_instance("Instance 1", distA, costsA, num_optima=1000, base_seed=123)
    outB = analyze_instance("Instance 2", distB, costsB, num_optima=1000, base_seed=5000)

    rows = []
    for out in [outA, outB]:
        for inst, mode, measure, r in out["correlations"]:
            rows.append({
                "Instance": inst,
                "Similarity mode": mode,
                "Measure": measure,
                "Pearson r": r
            })
    df = pd.DataFrame(rows)
    print("\n=== Correlation coefficients (Pearson r) ===")
    print(df.to_string(index=False))

    print("\n=== Good solutions used ===")
    print("Instance 1 best-of-1000 obj:", outA["best_of_1000"][0])
    print("Instance 1 best-of-1000 tour:", outA["best_of_1000"][1])
    print("Instance 1 very-good obj:", outA["very_good"][0])
    print("Instance 1 very-good tour:", outA["very_good"][1])

    print("\nInstance 2 best-of-1000 obj:", outB["best_of_1000"][0])
    print("Instance 2 best-of-1000 tour:", outB["best_of_1000"][1])
    print("Instance 2 very-good obj:", outB["very_good"][0])
    print("Instance 2 very-good tour:", outB["very_good"][1])


if __name__ == "__main__":
    main()

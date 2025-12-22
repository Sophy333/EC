import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ec_amoms

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
            dij = round_half_up(d)
            dist[i, j] = dij
            dist[j, i] = dij
    return dist

def read_instance_csv_with_coords(path: str, sep=";"):
    df = pd.read_csv(path, sep=sep, header=None)
    coords = df[[0, 1]].to_numpy(dtype=np.int32)
    costs  = df[2].to_numpy(dtype=np.int32)
    dist = build_distance_matrix(coords)
    return coords, dist, costs


def calc_cycle_length(dist: np.ndarray, tour) -> int:
    t = np.asarray(tour, dtype=int)
    m = len(t)
    return int(sum(dist[t[i], t[(i + 1) % m]] for i in range(m)))

def calc_obj(dist: np.ndarray, costs: np.ndarray, tour) -> int:
    t = np.asarray(tour, dtype=int)
    return calc_cycle_length(dist, t) + int(costs[t].sum())


def fmt_av_min_max(vals):
    a = np.asarray(vals, dtype=np.int64)
    av = int(round(a.mean()))
    mn = int(a.min())
    mx = int(a.max())
    s = lambda x: f"{x:,}".replace(",", " ")
    return f"{s(av)} ({s(mn)} – {s(mx)})"

def plot_solution(coords: np.ndarray, tour, title="Solution"):
    coords = np.asarray(coords)
    tour = np.asarray(tour, dtype=int)

    n = coords.shape[0]
    sel = np.zeros(n, dtype=bool)
    sel[tour] = True

    plt.figure(figsize=(8, 8))
    plt.scatter(coords[~sel, 0], coords[~sel, 1], s=10, alpha=0.35, label="Unchosen")
    plt.scatter(coords[sel, 0], coords[sel, 1], s=28, label="Chosen")

    path = coords[tour]
    closed = np.vstack([path, path[0]])
    plt.plot(closed[:, 0], closed[:, 1], linewidth=1.4, label="Cycle")

    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_amoms_20(dist, costs, time_limit_s, base_seed=1000, use_recomb=True):
    objs = []
    best_obj = None
    best_tour = None
    best_stats = None

    for r in range(20):
        seed = base_seed + 10000 * r
        tour, obj, stats = ec_amoms.run(dist, costs, time_limit_s, seed, use_recomb)
        tour = np.asarray(tour, dtype=int)
        obj = int(obj)

        objs.append(obj)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_tour = tour
            best_stats = stats

    return np.asarray(objs, dtype=np.int64), best_tour, int(best_obj), best_stats


def main():
    instances = [
        ("Instance 1", "TSPA.csv", 1000),
        ("Instance 2", "TSPB.csv", 5000),
    ]

    results = {}

    for name, path, seed in instances:
        coords, dist, costs = read_instance_csv_with_coords(path)

        avg_time, times, bests = ec_amoms.msls_avg_time(dist, costs, runs=20, ls_runs=200, seed=seed)

        print(f"\n{name}: avg MSLS time = {avg_time:.3f}s")
        print(f"{name}: MSLS best objs (first 5): {np.asarray(bests)[:5]}")

        # Run AMOMS for 20 runs with same time limit
        objs, best_tour, best_obj, best_stats = run_amoms_20(
            dist, costs, time_limit_s=float(avg_time),
            base_seed=seed + 999, use_recomb=True
        )

        results[name] = {
            "coords": coords, "dist": dist, "costs": costs,
            "time_limit": float(avg_time),
            "objs": objs,
            "best_tour": best_tour,
            "best_obj": best_obj,
            "best_stats": best_stats,
        }

    table = pd.DataFrame({
        "Method": ["AMOMS (final improved method)"],
        "Instance 1": [fmt_av_min_max(results["Instance 1"]["objs"])],
        "Instance 2": [fmt_av_min_max(results["Instance 2"]["objs"])],
    })

    print("\n=== Final results table (avg (min – max)) ===")
    print(table.to_string(index=False))

    for inst_name in ["Instance 1", "Instance 2"]:
        best_tour = results[inst_name]["best_tour"]
        best_obj  = results[inst_name]["best_obj"]
        dist = results[inst_name]["dist"]
        costs = results[inst_name]["costs"]

        recomputed = calc_obj(dist, costs, best_tour)
        print(f"\n{inst_name} best obj reported: {best_obj}, recomputed: {recomputed}")
        print(f"{inst_name} best tour nodes:")
        print(best_tour.tolist())

        coords = results[inst_name]["coords"]
        L = calc_cycle_length(dist, best_tour)
        C = int(costs[best_tour].sum())
        title = f"{inst_name} | AMOMS best | obj={best_obj} (len={L}, cost={C})"
        plot_solution(coords, best_tour, title=title)

        print(f"{inst_name} best stats:", dict(results[inst_name]["best_stats"]))

if __name__ == "__main__":
    main()

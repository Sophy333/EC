import math
import time
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


def summarize(objs):
    arr = np.asarray(objs, dtype=int)
    return float(arr.mean()), int(arr.min()), int(arr.max())


def fmt_av_min_max(objs):
    av, mn, mx = summarize(objs)

    def s(x): return f"{x:,}".replace(",", " ")

    return f"{s(int(round(av)))} ({s(mn)}, {s(mx)})"

def run_baseline_steepest(dist, costs, reps=200, base_seed=123, keep_best=True):
    objs = []
    best_obj = None
    best_tour = None
    best_seed = None

    t0 = time.time()
    for r in range(reps):
        seed = base_seed + r
        tour, obj, _ = ec_ls.steepest_full(dist, costs, seed=seed)
        obj = int(obj)
        objs.append(obj)

        if keep_best and (best_obj is None or obj < best_obj):
            best_obj = obj
            best_tour = np.asarray(tour, dtype=int)
            best_seed = seed

    return {
        "objs": objs,
        "time": time.time() - t0,
        "best_obj": best_obj,
        "best_tour": best_tour,
        "best_seed": best_seed,
    }


def run_candidate_steepest(dist, costs, k, reps=200, base_seed=123, keep_best=True):
    objs = []
    best_obj = None
    best_tour = None
    best_seed = None

    t0 = time.time()
    for r in range(reps):
        seed = base_seed + r
        tour, obj, _ = ec_ls.steepest_candidate(dist, costs, k=k, seed=seed)
        obj = int(obj)
        objs.append(obj)

        if keep_best and (best_obj is None or obj < best_obj):
            best_obj = obj
            best_tour = np.asarray(tour, dtype=int)
            best_seed = seed

    return {
        "objs": objs,
        "time": time.time() - t0,
        "best_obj": best_obj,
        "best_tour": best_tour,
        "best_seed": best_seed,
    }

def build_report_table(distA, costsA, distB, costsB, ks=(10, 5, 3), reps=200, base_seed=123):
    rows = []

    header_text = (
        "Steepest local search using\n"
        "candidate moves with\n"
        "Random starting solution\n"
        "With edge exchange"
    )

    # Candidate rows
    for idx, k in enumerate(ks):
        resA = run_candidate_steepest(distA, costsA, k=k, reps=reps, base_seed=base_seed, keep_best=False)
        resB = run_candidate_steepest(distB, costsB, k=k, reps=reps, base_seed=base_seed, keep_best=False)

        rows.append({
            "Method": header_text if idx == 0 else "",
            "k": f"k = {k}",
            "Instance 1": fmt_av_min_max(resA["objs"]),
            "Instance 2": fmt_av_min_max(resB["objs"]),
        })

    baseA = run_baseline_steepest(distA, costsA, reps=reps, base_seed=base_seed, keep_best=False)
    baseB = run_baseline_steepest(distB, costsB, reps=reps, base_seed=base_seed, keep_best=False)

    rows.append({
        "Method": (
            "Steepest local search with\n"
            "Random starting solution\n"
            "With edges exchange"
        ),
        "k": "",
        "Instance 1": fmt_av_min_max(baseA["objs"]),
        "Instance 2": fmt_av_min_max(baseB["objs"]),
    })

    return pd.DataFrame(rows)

def plot_solution(coords: np.ndarray, tour, title="Solution"):
    coords = np.asarray(coords)
    tour = np.asarray(tour, dtype=int)

    n = coords.shape[0]
    selected = np.zeros(n, dtype=bool)
    selected[tour] = True

    chosen_xy = coords[selected]
    unchosen_xy = coords[~selected]

    plt.figure(figsize=(7, 7))
    if len(unchosen_xy) > 0:
        plt.scatter(unchosen_xy[:, 0], unchosen_xy[:, 1], s=12, label="Unchosen")
    plt.scatter(chosen_xy[:, 0], chosen_xy[:, 1], s=20, label="Chosen")

    path_xy = coords[tour]
    closed = np.vstack([path_xy, path_xy[0]])
    plt.plot(closed[:, 0], closed[:, 1], linewidth=1.2, label="Cycle")

    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_best_solutions_for_instance(instance_name, coords, dist, costs, ks=(10, 5, 3), reps=200, base_seed=123):
    # Baseline best
    base = run_baseline_steepest(dist, costs, reps=reps, base_seed=base_seed, keep_best=True)
    plot_solution(
        coords, base["best_tour"],
        title=f"{instance_name} | Baseline steepest (best of {reps})\nobj={base['best_obj']} seed={base['best_seed']}"
    )

    # Candidate best for each k
    for k in ks:
        cand = run_candidate_steepest(dist, costs, k=k, reps=reps, base_seed=base_seed, keep_best=True)
        plot_solution(
            coords, cand["best_tour"],
            title=f"{instance_name} | Candidate steepest k={k} (best of {reps})\nobj={cand['best_obj']} seed={cand['best_seed']}"
        )


def main():
    reps = 200
    base_seed = 123
    ks = (10, 5, 3)

    print("Reading instances...")
    coordsA, distA, costsA = read_instance_csv_with_coords("TSPA.csv")
    coordsB, distB, costsB = read_instance_csv_with_coords("TSPB.csv")

    print("Running experiments for the report table...")
    table = build_report_table(distA, costsA, distB, costsB, ks=ks, reps=reps, base_seed=base_seed)

    print("\n=== Final results table ===")
    print(table.to_string(index=False))

    print("\nPlotting best solutions (baseline + candidates)...")
    plot_best_solutions_for_instance("TSPA", coordsA, distA, costsA, ks=ks, reps=reps, base_seed=base_seed)
    plot_best_solutions_for_instance("TSPB", coordsB, distB, costsB, ks=ks, reps=reps, base_seed=base_seed)


if __name__ == "__main__":
    main()

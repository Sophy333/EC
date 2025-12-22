import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ec_regret

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

def fmt_av_min_max(objs):
    a = np.asarray(objs, dtype=np.int64)
    av = int(round(a.mean()))
    mn = int(a.min())
    mx = int(a.max())
    s = lambda x: f"{x:,}".replace(",", " ")
    return f"{s(av)} ({s(mn)} – {s(mx)})"

def plot_solution(coords, tour, title):
    coords = np.asarray(coords)
    tour = np.asarray(tour, dtype=int)

    n = coords.shape[0]
    sel = np.zeros(n, dtype=bool)
    sel[tour] = True

    plt.figure(figsize=(7, 7))
    plt.scatter(coords[~sel, 0], coords[~sel, 1], s=10, alpha=0.35, label="Unchosen")
    plt.scatter(coords[sel, 0], coords[sel, 1], s=28, label="Chosen")

    path = coords[tour]
    closed = np.vstack([path, path[0]])
    plt.plot(closed[:, 0], closed[:, 1], linewidth=1.3, label="Cycle")

    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_instance(coords, dist, costs, seed=123, wR=1.0, wB=1.0):
    out = {}

    objs0, best0, best_obj0, reps0 = ec_regret.run200(dist, costs, 0, seed, wR, wB)
    out["Greedy 2-regret"] = {
        "objs": np.asarray(objs0, dtype=np.int64),
        "best_tour": np.asarray(best0, dtype=int),
        "best_obj": int(best_obj0),
        "reps": int(reps0),
    }

    objs1, best1, best_obj1, reps1 = ec_regret.run200(dist, costs, 1, seed, wR, wB)
    out[f"Weighted (2-regret + bestΔ) wR={wR}, wB={wB}"] = {
        "objs": np.asarray(objs1, dtype=np.int64),
        "best_tour": np.asarray(best1, dtype=int),
        "best_obj": int(best_obj1),
        "reps": int(reps1),
    }

    return out


def main():
    instances = [("Instance 1", "TSPA.csv"), ("Instance 2", "TSPB.csv")]

    wR, wB = 1.0, 1.0

    per = {}
    for inst_name, path in instances:
        coords, dist, costs = read_instance_csv_with_coords(path)
        per[inst_name] = (coords, dist, costs, run_instance(coords, dist, costs, seed=123, wR=wR, wB=wB))

    methods = list(per["Instance 1"][3].keys())
    table = pd.DataFrame({"Method": methods})
    table["Instance 1"] = [fmt_av_min_max(per["Instance 1"][3][m]["objs"]) for m in methods]
    table["Instance 2"] = [fmt_av_min_max(per["Instance 2"][3][m]["objs"]) for m in methods]

    print("\n=== Regret heuristics table (avg (min – max)) ===")
    print(table.to_string(index=False))

    for inst_name in ["Instance 1", "Instance 2"]:
        coords, dist, costs, res = per[inst_name]
        for m in methods:
            title = f"{inst_name} | {m} | best obj = {res[m]['best_obj']}"
            plot_solution(coords, res[m]["best_tour"], title)


if __name__ == "__main__":
    main()

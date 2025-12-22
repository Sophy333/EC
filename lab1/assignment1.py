import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ec_construct


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

def read_instance_csv_with_coords(path: str, sep=";"):
    import pandas as pd
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
    if np.any(~sel):
        plt.scatter(coords[~sel, 0], coords[~sel, 1], s=12, label="Unchosen")
    plt.scatter(coords[sel, 0], coords[sel, 1], s=22, label="Chosen")

    path = coords[tour]
    closed = np.vstack([path, path[0]])
    plt.plot(closed[:, 0], closed[:, 1], linewidth=1.2, label="Cycle")

    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_instance(coords, dist, costs, seed=123):
    methods = [
        (0, "Random solution"),
        (1, "Nearest neighbor (add at end)"),
        (2, "Nearest neighbor (insert anywhere in path)"),
        (3, "Greedy cycle (cheapest insertion)"),
    ]
    out = {}
    for mid, name in methods:
        objs, best_tour, best_obj, reps = ec_construct.run200(dist, costs, mid, seed)
        out[mid] = {
            "name": name,
            "objs": np.asarray(objs, dtype=np.int64),
            "best_tour": np.asarray(best_tour, dtype=int),
            "best_obj": int(best_obj),
            "reps": int(reps),
        }
    return out

def main():
    instances = [("Instance 1", "TSPA.csv"), ("Instance 2", "TSPB.csv")]
    per = {}

    for inst_name, path in instances:
        coords, dist, costs = read_instance_csv_with_coords(path)
        per[inst_name] = (coords, dist, costs, run_instance(coords, dist, costs, seed=123))

    # Table
    methods_order = [0, 1, 2, 3]
    table = pd.DataFrame({"Method": [per["Instance 1"][3][m]["name"] for m in methods_order]})
    table["Instance 1"] = [fmt_av_min_max(per["Instance 1"][3][m]["objs"]) for m in methods_order]
    table["Instance 2"] = [fmt_av_min_max(per["Instance 2"][3][m]["objs"]) for m in methods_order]

    print("\n=== Construction heuristics table (avg (min – max)) ===")
    print(table.to_string(index=False))

    # Best plots
    for inst_name in ["Instance 1", "Instance 2"]:
        coords, dist, costs, res = per[inst_name]
        for m in methods_order:
            title = f"{inst_name} | {res[m]['name']} | best obj = {res[m]['best_obj']}"
            plot_solution(coords, res[m]["best_tour"], title)

if __name__ == "__main__":
    main()

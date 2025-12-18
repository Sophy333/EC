import math
import time
import numpy as np
import pandas as pd
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

def read_instance_csv(path: str, sep=";"):
    df = pd.read_csv(path, sep=sep, header=None)
    coords = df[[0, 1]].to_numpy(dtype=np.int32)
    costs  = df[2].to_numpy(dtype=np.int32)
    dist = build_distance_matrix(coords)
    return dist, costs


def summarize(objs):
    arr = np.asarray(objs, dtype=int)
    return float(arr.mean()), int(arr.min()), int(arr.max())

def fmt_av_min_max(objs):
    av, mn, mx = summarize(objs)
    def s(x): return f"{x:,}".replace(",", " ")
    return f"{s(int(round(av)))} ({s(mn)}, {s(mx)})"


def run_method(method_fn, dist, costs, reps=200, base_seed=123):
    objs = []
    t0 = time.time()
    for r in range(reps):
        seed = base_seed + r
        _, obj, _ = method_fn(dist, costs, seed=seed)
        objs.append(int(obj))
    return objs, time.time() - t0

def build_table(distA, costsA, distB, costsB, reps=200, base_seed=123):
    def run_method(method_fn, dist, costs):
        objs = []
        for r in range(reps):
            seed = base_seed + r
            _, obj, _ = method_fn(dist, costs, seed=seed)
            objs.append(int(obj))
        return objs

    rows = []

    objsA = run_method(ec_ls.steepest_lm, distA, costsA)
    objsB = run_method(ec_ls.steepest_lm, distB, costsB)

    rows.append({
        "Method": "Steepest local search using\nlist of improving moves (LM)\nRandom starting solution",
        "Instance 1": fmt_av_min_max(objsA),
        "Instance 2": fmt_av_min_max(objsB),
    })

    return pd.DataFrame(rows)


def main():
    distA, costsA = read_instance_csv("TSPA.csv")
    distB, costsB = read_instance_csv("TSPB.csv")

    table = build_table(distA, costsA, distB, costsB, reps=200, base_seed=123)

    print("\n=== Results table ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

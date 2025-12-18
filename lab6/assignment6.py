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
    return coords, dist, costs

def summarize(objs):
    a = np.asarray(objs, dtype=int)
    return float(a.mean()), int(a.min()), int(a.max())


def fmt_av_min_max(objs):
    av, mn, mx = summarize(objs)
    def s(x): return f"{x:,}".replace(",", " ")
    return f"{s(int(round(av)))} ({s(mn)} – {s(mx)})"

def run_ls_random(dist, costs, seed):
    tour, obj, _ = ec_ls.steepest_full(dist, costs, seed=int(seed))
    return np.asarray(tour, dtype=int), int(obj)


def run_ls_from_tour(dist, costs, tour):
    tour2, obj, _ = ec_ls.steepest_from_tour(dist, costs, np.asarray(tour, dtype=np.int32))
    return np.asarray(tour2, dtype=int), int(obj)


def random_tour(n, m, rng: np.random.Generator):
    tour = rng.permutation(n)[:m].astype(int).tolist()
    rng.shuffle(tour)
    return tour


def perturb_tour(tour, n, rng: np.random.Generator, strength=None):

    t = list(map(int, tour))
    m = len(t)
    sel = set(t)

    if strength is None:
        strength = max(2, m // 10)

    for _ in range(strength):
        if rng.random() < 0.5:
            pos = int(rng.integers(0, m))
            out = t[pos]
            v = int(rng.integers(0, n))
            while v in sel:
                v = int(rng.integers(0, n))
            t[pos] = v
            sel.remove(out)
            sel.add(v)
        else:
            i = int(rng.integers(0, m))
            j = int(rng.integers(0, m))
            if i > j:
                i, j = j, i
            if j - i >= 2:
                t[i:j+1] = reversed(t[i:j+1])

    # safety (should not happen, but keep robust)
    if len(set(t)) != m:
        t = list(dict.fromkeys(t))
        while len(t) < m:
            v = int(rng.integers(0, n))
            if v not in t:
                t.append(v)
    return t


def msls_one_run(dist, costs, msls_iters=200, seed=123):
    best_obj = None
    best_tour = None
    t0 = time.time()
    for i in range(msls_iters):
        tour, obj = run_ls_random(dist, costs, seed + i)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_tour = tour
    return best_tour, int(best_obj), (time.time() - t0)


def msls_experiment(dist, costs, runs=20, msls_iters=200, base_seed=1000):
    objs = []
    times = []
    best_obj = None
    best_tour = None

    for r in range(runs):
        tour, obj, t = msls_one_run(dist, costs, msls_iters=msls_iters, seed=base_seed + 10000*r)
        objs.append(int(obj))
        times.append(float(t))
        if best_obj is None or obj < best_obj:
            best_obj = int(obj)
            best_tour = tour

    return {
        "objs": objs,
        "times": times,
        "avg_time": float(np.mean(times)),
        "best_obj": int(best_obj),
        "best_tour": best_tour,
    }


def ils_one_run(dist, costs, time_limit_s, seed=777, strength=None):
    n = len(costs)
    m = (n + 1) // 2
    rng = np.random.default_rng(seed)

    start = random_tour(n, m, rng)
    cur_tour, cur_obj = run_ls_from_tour(dist, costs, start)

    best_tour = cur_tour.copy()
    best_obj = int(cur_obj)

    ls_runs = 1
    t0 = time.time()

    while (time.time() - t0) < time_limit_s:
        pert = perturb_tour(cur_tour, n, rng, strength=strength)
        new_tour, new_obj = run_ls_from_tour(dist, costs, pert)
        ls_runs += 1

        cur_tour, cur_obj = new_tour, new_obj

        if new_obj < best_obj:
            best_obj = int(new_obj)
            best_tour = new_tour.copy()

    return best_tour, best_obj, ls_runs


def ils_experiment(dist, costs, runs=20, time_limit_s=1.0, base_seed=2000, strength=None):
    objs = []
    ls_counts = []
    best_obj = None
    best_tour = None

    for r in range(runs):
        tour, obj, cnt = ils_one_run(
            dist, costs,
            time_limit_s=time_limit_s,
            seed=base_seed + 10000*r,
            strength=strength
        )
        objs.append(int(obj))
        ls_counts.append(int(cnt))

        if best_obj is None or obj < best_obj:
            best_obj = int(obj)
            best_tour = tour

    return {
        "objs": objs,
        "ls_counts": ls_counts,
        "best_obj": int(best_obj),
        "best_tour": best_tour,
    }


def build_results_table(msls_A, msls_B, ils_A, ils_B):
    return pd.DataFrame([
        {
            "Method": "MSLS (20 runs)\nEach run: best of 200 steepest LS",
            "Instance 1": fmt_av_min_max(msls_A["objs"]),
            "Instance 2": fmt_av_min_max(msls_B["objs"]),
        },
        {
            "Method": "ILS (20 runs)\nStop time = avg MSLS time",
            "Instance 1": fmt_av_min_max(ils_A["objs"]),
            "Instance 2": fmt_av_min_max(ils_B["objs"]),
        },
    ])


def build_ils_counts_table(ils_A, ils_B):
    return pd.DataFrame([{
        "Method": "ILS: # of basic LS runs (per ILS run)",
        "Instance 1": fmt_av_min_max(ils_A["ls_counts"]),
        "Instance 2": fmt_av_min_max(ils_B["ls_counts"]),
    }])


def main():
    RUNS = 20
    MSLS_ITERS = 200

    ILS_STRENGTH = None

    _, distA, costsA = read_instance_csv("TSPA.csv")
    _, distB, costsB = read_instance_csv("TSPB.csv")

    print("Running MSLS...")
    msls_A = msls_experiment(distA, costsA, runs=RUNS, msls_iters=MSLS_ITERS, base_seed=1000)
    msls_B = msls_experiment(distB, costsB, runs=RUNS, msls_iters=MSLS_ITERS, base_seed=3000)

    time_limit_A = msls_A["avg_time"]
    time_limit_B = msls_B["avg_time"]

    print(f"\nAvg MSLS time limits for ILS: Instance1={time_limit_A:.3f}s, Instance2={time_limit_B:.3f}s")

    print("\nRunning ILS...")
    ils_A = ils_experiment(distA, costsA, runs=RUNS, time_limit_s=time_limit_A, base_seed=5000, strength=ILS_STRENGTH)
    ils_B = ils_experiment(distB, costsB, runs=RUNS, time_limit_s=time_limit_B, base_seed=7000, strength=ILS_STRENGTH)

    table = build_results_table(msls_A, msls_B, ils_A, ils_B)
    print("\n=== Results table (avg (min – max)) ===")
    print(table.to_string(index=False))

    counts = build_ils_counts_table(ils_A, ils_B)
    print("\n=== ILS: number of basic LS runs (avg (min – max)) ===")
    print(counts.to_string(index=False))

    print("\n=== Best solutions (node lists) ===")
    print("MSLS best Instance 1:", msls_A["best_obj"])
    print(msls_A["best_tour"].tolist())
    print("\nMSLS best Instance 2:", msls_B["best_obj"])
    print(msls_B["best_tour"].tolist())

    print("\nILS best Instance 1:", ils_A["best_obj"])
    print(ils_A["best_tour"].tolist())
    print("\nILS best Instance 2:", ils_B["best_obj"])
    print(ils_B["best_tour"].tolist())


if __name__ == "__main__":
    main()

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


def summarize(vals):
    a = np.asarray(vals, dtype=int)
    return float(a.mean()), int(a.min()), int(a.max())

def fmt_av_min_max(vals):
    av, mn, mx = summarize(vals)
    def s(x): return f"{x:,}".replace(",", " ")
    return f"{s(int(round(av)))} ({s(mn)} – {s(mx)})"


def cycle_length(dist, tour):
    t = np.asarray(tour, dtype=int)
    m = len(t)
    return int(sum(dist[t[i], t[(i + 1) % m]] for i in range(m)))

def cost_sum(costs, tour):
    t = np.asarray(tour, dtype=int)
    return int(costs[t].sum())

def objective(dist, costs, tour):
    return cycle_length(dist, tour) + cost_sum(costs, tour)


def ls_from_tour(dist, costs, tour):
    tour2, obj, _ = ec_ls.steepest_from_tour(dist, costs, np.asarray(tour, dtype=np.int32))
    return np.asarray(tour2, dtype=int), int(obj)

def ls_random(dist, costs, seed):
    tour2, obj, _ = ec_ls.steepest_full(dist, costs, seed=int(seed))
    return np.asarray(tour2, dtype=int), int(obj)


def random_tour(n, m, rng: np.random.Generator):
    t = rng.permutation(n)[:m].astype(int).tolist()
    rng.shuffle(t)
    return t

def destroy_nodes_biased(dist, costs, tour, rng: np.random.Generator, frac=0.45, power=2.0):

    t = list(map(int, tour))
    m = len(t)
    k = int(round(frac * m))
    k = max(1, min(k, m - 2))  # keep at least 2 nodes

    bad = np.zeros(m, dtype=np.float64)
    for i in range(m):
        prev = t[(i - 1) % m]
        cur  = t[i]
        nxt  = t[(i + 1) % m]
        bad[i] = float(dist[prev, cur] + dist[cur, nxt] + costs[cur])

    w = bad - bad.min()
    w = w + 1e-9
    w = np.power(w, power)
    p = w / w.sum()

    remove_idx = rng.choice(np.arange(m), size=k, replace=False, p=p)
    remove_idx = set(int(i) for i in remove_idx)

    remaining = [t[i] for i in range(m) if i not in remove_idx]
    return remaining

def destroy_subpath(tour, rng: np.random.Generator, frac=0.45):

    t = list(map(int, tour))
    m = len(t)
    k = int(round(frac * m))
    k = max(1, min(k, m - 2))

    start = int(rng.integers(0, m))
    remove = set((start + i) % m for i in range(k))

    remaining = [t[i] for i in range(m) if i not in remove]
    return remaining


def best_insertion_delta(dist, costs, tour, v):

    m = len(tour)
    best = None
    second = None
    best_pos = None

    for i in range(m):
        a = tour[i]
        b = tour[(i + 1) % m]
        d = int(dist[a, v] + dist[v, b] - dist[a, b] + costs[v])

        if best is None or d < best:
            second = best
            best = d
            best_pos = i
        elif second is None or d < second:
            second = d

    if second is None:
        second = best
    return best, second, best_pos

def repair_regret2(dist, costs, n, m_target, partial_tour, rng: np.random.Generator, tie_p=0.35):

    tour = list(dict.fromkeys(map(int, partial_tour)))
    selected = set(tour)

    if len(tour) < 2:
        pool = [i for i in range(n) if i not in selected]
        rng.shuffle(pool)
        while len(tour) < 2:
            tour.append(pool.pop())
            selected = set(tour)

    while len(tour) < m_target:
        best_v = None
        best_regret = None
        best_best = None
        best_pos = None

        for v in range(n):
            if v in selected:
                continue

            b1, b2, pos = best_insertion_delta(dist, costs, tour, v)
            regret = b2 - b1

            choose = False
            if best_v is None:
                choose = True
            elif regret > best_regret:
                choose = True
            elif regret == best_regret and b1 < best_best:
                choose = True
            elif regret == best_regret and b1 == best_best and rng.random() < tie_p:
                choose = True

            if choose:
                best_v = v
                best_regret = regret
                best_best = b1
                best_pos = pos

        tour.insert(best_pos + 1, best_v)
        selected.add(best_v)

    return tour


def msls_avg_time(dist, costs, runs=20, msls_iters=200, base_seed=1000):
    times = []
    for r in range(runs):
        seed = base_seed + 10000 * r
        t0 = time.time()
        best = None
        for i in range(msls_iters):
            _, obj = ls_random(dist, costs, seed + i)
            if best is None or obj < best:
                best = obj
        times.append(time.time() - t0)
    return float(np.mean(times))


def lns_one_run(
    dist, costs,
    seed,
    time_limit_s,
    use_ls_after_repair: bool,
    destroy_frac=0.45,
    destroy_mode="biased",   # "biased" or "subpath"
):
    
    n = len(costs)
    m = (n + 1) // 2
    rng = np.random.default_rng(seed)

    # initial random -> LS (mandatory)
    x0 = random_tour(n, m, rng)
    x, fx = ls_from_tour(dist, costs, x0)

    best_tour = x.copy()
    best_obj = int(fx)

    iters = 0
    t0 = time.time()

    while (time.time() - t0) < time_limit_s:
        iters += 1

        # Destroy
        if destroy_mode == "subpath":
            partial = destroy_subpath(x, rng, frac=destroy_frac)
        else:
            partial = destroy_nodes_biased(dist, costs, x, rng, frac=destroy_frac, power=2.0)

        # Repair
        y = repair_regret2(dist, costs, n, m, partial, rng, tie_p=0.35)

        # Optional LS after repair
        if use_ls_after_repair:
            y, fy = ls_from_tour(dist, costs, y)
        else:
            fy = objective(dist, costs, y)
        x, fx = np.asarray(y, dtype=int), int(fy)

        # Track global best
        if fx < best_obj:
            best_obj = int(fx)
            best_tour = x.copy()

    return best_tour, best_obj, iters


def lns_experiment(
    dist, costs,
    runs=20,
    time_limit_s=1.0,
    base_seed=5000,
    use_ls_after_repair=False,
    destroy_frac=0.45,
    destroy_mode="biased",
):
    objs = []
    iters_list = []
    best_obj = None
    best_tour = None

    for r in range(runs):
        tour, obj, iters = lns_one_run(
            dist, costs,
            seed=base_seed + 10000 * r,
            time_limit_s=time_limit_s,
            use_ls_after_repair=use_ls_after_repair,
            destroy_frac=destroy_frac,
            destroy_mode=destroy_mode,
        )
        objs.append(int(obj))
        iters_list.append(int(iters))
        if best_obj is None or obj < best_obj:
            best_obj = int(obj)
            best_tour = np.asarray(tour, dtype=int)

    return {
        "objs": objs,
        "iters": iters_list,
        "best_obj": int(best_obj),
        "best_tour": best_tour
    }

def build_results_table(A_no, B_no, A_ls, B_ls):
    return pd.DataFrame([
        {
            "Method": "Large neighborhood search\n(without local search)",
            "Instance 1": fmt_av_min_max(A_no["objs"]),
            "Instance 2": fmt_av_min_max(B_no["objs"]),
        },
        {
            "Method": "Large neighborhood search\n(with local search)",
            "Instance 1": fmt_av_min_max(A_ls["objs"]),
            "Instance 2": fmt_av_min_max(B_ls["objs"]),
        },
    ])

def build_iters_table(A_no, B_no, A_ls, B_ls):
    return pd.DataFrame([
        {
            "Method": "LNS iterations\n(without local search)",
            "Instance 1": fmt_av_min_max(A_no["iters"]),
            "Instance 2": fmt_av_min_max(B_no["iters"]),
        },
        {
            "Method": "LNS iterations\n(with local search)",
            "Instance 1": fmt_av_min_max(A_ls["iters"]),
            "Instance 2": fmt_av_min_max(B_ls["iters"]),
        },
    ])


def main():
    RUNS = 20

    # Stronger destroy helps a lot
    DESTROY_FRAC = 0.5           
    DESTROY_MODE = "subpath"       

    TIME_LIMITS = None  # {"TSPA": 1.23, "TSPB": 0.98}

    _, distA, costsA = read_instance_csv("TSPA.csv")
    _, distB, costsB = read_instance_csv("TSPB.csv")

    # time limits from MSLS avg time
    if TIME_LIMITS is None:
        print("Measuring avg MSLS time limits...")
        tlA = msls_avg_time(distA, costsA, runs=20, msls_iters=200, base_seed=1000)
        tlB = msls_avg_time(distB, costsB, runs=20, msls_iters=200, base_seed=3000)
        print(f"Avg MSLS time: Instance1={tlA:.3f}s, Instance2={tlB:.3f}s")
    else:
        tlA = float(TIME_LIMITS["TSPA"])
        tlB = float(TIME_LIMITS["TSPB"])

    print("\nRunning LNS WITHOUT local search after repair...")
    A_no = lns_experiment(distA, costsA, runs=RUNS, time_limit_s=tlA, base_seed=5000,
                          use_ls_after_repair=False, destroy_frac=DESTROY_FRAC, destroy_mode=DESTROY_MODE)
    B_no = lns_experiment(distB, costsB, runs=RUNS, time_limit_s=tlB, base_seed=7000,
                          use_ls_after_repair=False, destroy_frac=DESTROY_FRAC, destroy_mode=DESTROY_MODE)

    print("Running LNS WITH local search after repair...")
    A_ls = lns_experiment(distA, costsA, runs=RUNS, time_limit_s=tlA, base_seed=9000,
                          use_ls_after_repair=True, destroy_frac=DESTROY_FRAC, destroy_mode=DESTROY_MODE)
    B_ls = lns_experiment(distB, costsB, runs=RUNS, time_limit_s=tlB, base_seed=11000,
                          use_ls_after_repair=True, destroy_frac=DESTROY_FRAC, destroy_mode=DESTROY_MODE)

    # Tables
    table = build_results_table(A_no, B_no, A_ls, B_ls)
    print("\n=== Results table (avg (min – max)) ===")
    print(table.to_string(index=False))

    it_table = build_iters_table(A_no, B_no, A_ls, B_ls)
    print("\n=== LNS iterations table (avg (min – max)) ===")
    print(it_table.to_string(index=False))

    # Best solutions
    print("\n=== Best solutions (node lists) ===")
    print("LNS (without LS) best Instance 1:", A_no["best_obj"])
    print(A_no["best_tour"].tolist())
    print("\nLNS (without LS) best Instance 2:", B_no["best_obj"])
    print(B_no["best_tour"].tolist())

    print("\nLNS (with LS) best Instance 1:", A_ls["best_obj"])
    print(A_ls["best_tour"].tolist())
    print("\nLNS (with LS) best Instance 2:", B_ls["best_obj"])
    print(B_ls["best_tour"].tolist())


if __name__ == "__main__":
    main()

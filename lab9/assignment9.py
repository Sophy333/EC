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
    costs = df[2].to_numpy(dtype=np.int32)
    dist = build_distance_matrix(coords)
    return coords, dist, costs

def summarize(vals):
    a = np.asarray(vals, dtype=int)
    return float(a.mean()), int(a.min()), int(a.max())


def fmt_av_min_max(vals):
    av, mn, mx = summarize(vals)

    def s(x):
        return f"{x:,}".replace(",", " ")

    return f"{s(int(round(av)))} ({s(mn)} – {s(mx)})"

def ls_random(dist, costs, seed):
    tour, obj, _ = ec_ls.steepest_full(dist, costs, int(seed))
    return np.asarray(tour, dtype=int), int(obj)


def ls_from_tour(dist, costs, tour):
    tour2, obj, _ = ec_ls.steepest_from_tour(dist, costs, np.asarray(tour, dtype=np.int32))
    return np.asarray(tour2, dtype=int), int(obj)

def random_tour(n, m, rng: np.random.Generator):
    t = rng.permutation(n)[:m].astype(int).tolist()
    rng.shuffle(t)
    return t


def msls_one_run(dist, costs, msls_iters=200, seed=123):
    best_obj = None
    best_tour = None
    t0 = time.time()
    for i in range(msls_iters):
        tour, obj = ls_random(dist, costs, seed + i)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_tour = tour
    return best_tour, int(best_obj), (time.time() - t0)


def msls_experiment(dist, costs, runs=20, msls_iters=200, base_seed=1000):
    objs, times = [], []
    best_obj, best_tour = None, None
    for r in range(runs):
        tour, obj, t = msls_one_run(dist, costs, msls_iters=msls_iters, seed=base_seed + 10000 * r)
        objs.append(obj)
        times.append(t)
        if best_obj is None or obj < best_obj:
            best_obj, best_tour = obj, tour
    return {
        "objs": objs,
        "times": times,
        "avg_time": float(np.mean(times)),
        "best_obj": int(best_obj),
        "best_tour": best_tour,
    }

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
                t[i : j + 1] = reversed(t[i : j + 1])

    if len(set(t)) != m:
        t = list(dict.fromkeys(t))
        while len(t) < m:
            v = int(rng.integers(0, n))
            if v not in t:
                t.append(v)
    return t


def ils_one_run(dist, costs, time_limit_s, seed=777):
    n = len(costs)
    m = (n + 1) // 2
    rng = np.random.default_rng(seed)

    start = random_tour(n, m, rng)
    cur, cur_obj = ls_from_tour(dist, costs, start)
    best_tour = cur.copy()
    best_obj = int(cur_obj)
    ls_runs = 1

    t0 = time.time()
    while (time.time() - t0) < time_limit_s:
        pert = perturb_tour(cur, n, rng)
        cur, cur_obj = ls_from_tour(dist, costs, pert)
        ls_runs += 1
        if cur_obj < best_obj:
            best_obj = int(cur_obj)
            best_tour = cur.copy()

    return best_tour, best_obj, ls_runs


def ils_experiment(dist, costs, runs=20, time_limit_s=1.0, base_seed=2000):
    objs, counts = [], []
    best_obj, best_tour = None, None
    for r in range(runs):
        tour, obj, cnt = ils_one_run(dist, costs, time_limit_s, seed=base_seed + 10000 * r)
        objs.append(obj)
        counts.append(cnt)
        if best_obj is None or obj < best_obj:
            best_obj, best_tour = obj, tour
    return {"objs": objs, "counts": counts, "best_obj": int(best_obj), "best_tour": best_tour}


def objective_cycle(dist, costs, tour):
    t = np.asarray(tour, dtype=int)
    m = len(t)
    length = int(sum(dist[t[i], t[(i + 1) % m]] for i in range(m)))
    cs = int(costs[t].sum())
    return length + cs


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


def repair_regret2(dist, costs, n, m_target, partial_tour, rng: np.random.Generator, tie_p=0.25):
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
        best_reg = None
        best_b1 = None
        best_pos = None

        for v in range(n):
            if v in selected:
                continue
            b1, b2, pos = best_insertion_delta(dist, costs, tour, v)
            reg = b2 - b1

            choose = False
            if best_v is None:
                choose = True
            elif reg > best_reg:
                choose = True
            elif reg == best_reg and b1 < best_b1:
                choose = True
            elif reg == best_reg and b1 == best_b1 and rng.random() < tie_p:
                choose = True

            if choose:
                best_v, best_reg, best_b1, best_pos = v, reg, b1, pos

        tour.insert(best_pos + 1, best_v)
        selected.add(best_v)

    return tour


def destroy_remove_frac(tour, rng, frac=0.30):
    t = list(map(int, tour))
    m = len(t)
    k = max(1, min(m - 2, int(round(frac * m))))
    idx = rng.choice(np.arange(m), size=k, replace=False)
    rem = set(int(i) for i in idx)
    return [t[i] for i in range(m) if i not in rem]


def lns_one_run(dist, costs, time_limit_s, seed=333, use_ls_after_repair=True, destroy_frac=0.30):
    n = len(costs)
    m = (n + 1) // 2
    rng = np.random.default_rng(seed)

    x0 = random_tour(n, m, rng)
    x, fx = ls_from_tour(dist, costs, x0)
    best_tour = x.copy()
    best_obj = int(fx)

    t0 = time.time()
    while (time.time() - t0) < time_limit_s:
        partial = destroy_remove_frac(x, rng, frac=destroy_frac)
        y = repair_regret2(dist, costs, n, m, partial, rng)
        if use_ls_after_repair:
            y, fy = ls_from_tour(dist, costs, y)
        else:
            fy = objective_cycle(dist, costs, y)

        if fy < fx:
            x, fx = y, int(fy)
            if fx < best_obj:
                best_obj = int(fx)
                best_tour = np.asarray(x, dtype=int).copy()

    return best_tour, best_obj


def lns_experiment(dist, costs, runs=20, time_limit_s=1.0, base_seed=5000, use_ls_after_repair=True):
    objs = []
    best_obj, best_tour = None, None
    for r in range(runs):
        tour, obj = lns_one_run(
            dist,
            costs,
            time_limit_s,
            seed=base_seed + 10000 * r,
            use_ls_after_repair=use_ls_after_repair,
            destroy_frac=0.30,
        )
        objs.append(obj)
        if best_obj is None or obj < best_obj:
            best_obj, best_tour = obj, tour
    return {"objs": objs, "best_obj": int(best_obj), "best_tour": best_tour}


def hea_cpp_experiment(
    dist,
    costs,
    runs,
    time_limit_s,
    base_seed,
    recomb_type=1,
    p_ls=0.6,
    tourn_size=5,
    pop_size=20,
):
    objs, iters_list = [], []
    best_obj, best_tour = None, None
    for r in range(runs):
        tour, obj, iters = ec_ls.hea_run(
            dist,
            costs,
            float(time_limit_s),
            int(base_seed + 10000 * r),
            int(recomb_type),      # 1=op1, 2=op2
            float(p_ls),           # p_ls=0.0 -> NO LS on offspring
            int(tourn_size),
            int(pop_size),
        )
        obj = int(obj)
        tour = np.asarray(tour, dtype=int)
        objs.append(obj)
        iters_list.append(int(iters))
        if best_obj is None or obj < best_obj:
            best_obj, best_tour = obj, tour
    return {"objs": objs, "iters": iters_list, "best_obj": int(best_obj), "best_tour": best_tour}

def run_all_methods_for_instance(dist, costs, base_seed):
    RUNS = 20
    MSLS_ITERS = 200

    msls = msls_experiment(dist, costs, runs=RUNS, msls_iters=MSLS_ITERS, base_seed=base_seed)
    tl = msls["avg_time"]

    ils = ils_experiment(dist, costs, runs=RUNS, time_limit_s=tl, base_seed=base_seed + 2000)
    lns = lns_experiment(dist, costs, runs=RUNS, time_limit_s=tl, base_seed=base_seed + 5000, use_ls_after_repair=True)

    # HEA Op1 
    hea_op1 = hea_cpp_experiment(
        dist, costs, runs=RUNS, time_limit_s=tl, base_seed=base_seed + 9000,
        recomb_type=1, p_ls=0.6, tourn_size=5, pop_size=20
    )

    # HEA Op2 
    hea_op2 = hea_cpp_experiment(
        dist, costs, runs=RUNS, time_limit_s=tl, base_seed=base_seed + 13000,
        recomb_type=2, p_ls=0.8, tourn_size=5, pop_size=20
    )

    # HEA Op2 
    hea_op2_no_ls = hea_cpp_experiment(
        dist, costs, runs=RUNS, time_limit_s=tl, base_seed=base_seed + 17000,
        recomb_type=2, p_ls=0.0, tourn_size=5, pop_size=20
    )

    return {
        "time_limit": tl,
        "MSLS": msls,
        "ILS": ils,
        "LNS": lns,
        "HEA_op1": hea_op1,
        "HEA_op2": hea_op2,
        "HEA_op2_no_ls": hea_op2_no_ls,
    }


def build_results_table(resA, resB):
    rows = [
        {
            "Method": "MSLS (20 runs)\nBest of 200 steepest LS",
            "Instance 1": fmt_av_min_max(resA["MSLS"]["objs"]),
            "Instance 2": fmt_av_min_max(resB["MSLS"]["objs"]),
        },
        {
            "Method": "ILS (20 runs)\nStop time = avg MSLS time",
            "Instance 1": fmt_av_min_max(resA["ILS"]["objs"]),
            "Instance 2": fmt_av_min_max(resB["ILS"]["objs"]),
        },
        {
            "Method": "LNS (20 runs)\nStop time = avg MSLS time",
            "Instance 1": fmt_av_min_max(resA["LNS"]["objs"]),
            "Instance 2": fmt_av_min_max(resB["LNS"]["objs"]),
        },
        {
            "Method": "HEA (Op1, C++)\nStop time = avg MSLS time",
            "Instance 1": fmt_av_min_max(resA["HEA_op1"]["objs"]),
            "Instance 2": fmt_av_min_max(resB["HEA_op1"]["objs"]),
        },
        {
            "Method": "HEA (Op2, C++)\nStop time = avg MSLS time\nLS on offspring",
            "Instance 1": fmt_av_min_max(resA["HEA_op2"]["objs"]),
            "Instance 2": fmt_av_min_max(resB["HEA_op2"]["objs"]),
        },
        {
            "Method": "HEA (Op2, C++)\nStop time = avg MSLS time\nNO LS on offspring",
            "Instance 1": fmt_av_min_max(resA["HEA_op2_no_ls"]["objs"]),
            "Instance 2": fmt_av_min_max(resB["HEA_op2_no_ls"]["objs"]),
        },
    ]
    return pd.DataFrame(rows)


def main():
    _, distA, costsA = read_instance_csv("TSPA.csv")
    _, distB, costsB = read_instance_csv("TSPB.csv")

    print("Running comparison (MSLS / ILS / LNS / HEA(C++))...\n")
    resA = run_all_methods_for_instance(distA, costsA, base_seed=1000)
    resB = run_all_methods_for_instance(distB, costsB, base_seed=5000)

    print(
        f"Time limits (avg MSLS time): "
        f"Instance1={resA['time_limit']:.3f}s, Instance2={resB['time_limit']:.3f}s"
    )

    table = build_results_table(resA, resB)
    print("\n=== Results table (avg (min – max)) ===")
    print(table.to_string(index=False))

    print("\n=== Best solutions (node lists) ===")
    for inst_name, res in [("Instance 1", resA), ("Instance 2", resB)]:
        print(f"\n{inst_name}:")
        for key in ["MSLS", "ILS", "LNS", "HEA_op1", "HEA_op2", "HEA_op2_no_ls"]:
            print(f"{key} best obj = {res[key]['best_obj']}")
            print(res[key]["best_tour"].tolist())


if __name__ == "__main__":
    main()

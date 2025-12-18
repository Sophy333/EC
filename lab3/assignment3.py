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
            dij = round_half_up(d)
            dist[i, j] = dij
            dist[j, i] = dij
    return dist


def read_instance_csv_with_coords(path: str, sep: str = ";"):
    df = pd.read_csv(path, sep=sep, header=None)
    coords = df[[0, 1]].to_numpy(dtype=np.int32)
    costs = df[2].to_numpy(dtype=np.int32)
    dist = build_distance_matrix(coords)
    return coords, dist, costs

def calc_cycle_length(dist: np.ndarray, tour) -> int:
    tour = np.asarray(tour, dtype=int)
    m = len(tour)
    return sum(int(dist[tour[i], tour[(i + 1) % m]]) for i in range(m))


def calc_cost_sum(costs: np.ndarray, tour) -> int:
    tour = np.asarray(tour, dtype=int)
    return int(costs[tour].sum())


def obj_parts(dist: np.ndarray, costs: np.ndarray, tour):
    length = calc_cycle_length(dist, tour)
    cs = calc_cost_sum(costs, tour)
    return length, cs, length + cs


def method_configs():
    cfgs = []
    for search_type in [0, 1]:
        for intra_type in [0, 1]:
            for start_type in [0, 1]:
                s = "Steepest" if search_type == 0 else "Greedy"
                intra = "Intra: node-swap" if intra_type == 0 else "Intra: 2-opt"
                st = "Start: random" if start_type == 0 else "Start: greedy-regret"
                cfgs.append(
                    {
                        "name": f"{s} | {intra} | {st}",
                        "search_type": search_type,
                        "intra_type": intra_type,
                        "start_type": start_type,
                    }
                )
    return sorted(cfgs, key=lambda d: d["name"])


def run_method(dist: np.ndarray, costs: np.ndarray, cfg: dict, reps=200, base_seed=123, store_all=True):
    n = len(costs)
    m = (n + 1) // 2

    objs = []
    best_obj = None
    best_tour = None
    best_seed_info = None

    if cfg["start_type"] == 0:
        for r in range(reps):
            seed = base_seed + r
            tour, obj, _ = ec_ls.local_search(
                dist, costs, m,
                cfg["search_type"], cfg["intra_type"], cfg["start_type"],
                seed, -1
            )
            obj = int(obj)
            objs.append(obj)
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_tour = np.asarray(tour, dtype=int)
                best_seed_info = ("random_start", seed)
    else:
        for start_node in range(min(reps, n)):
            seed = base_seed + start_node
            tour, obj, _ = ec_ls.local_search(
                dist, costs, m,
                cfg["search_type"], cfg["intra_type"], cfg["start_type"],
                seed, start_node
            )
            obj = int(obj)
            objs.append(obj)
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_tour = np.asarray(tour, dtype=int)
                best_seed_info = ("start_node", start_node)

    row = {
        "Method": cfg["name"],
        "avg": float(np.mean(objs)),
        "min": int(np.min(objs)),
        "max": int(np.max(objs)),
        "best_obj": int(best_obj),
        "best_tour": best_tour,
        "best_seed_info": best_seed_info,
    }
    if store_all:
        row["objs"] = objs
    return row


def run_instance(coords, dist, costs, reps=200, base_seed=123, store_all=True):
    rows = []
    for cfg in method_configs():
        rows.append(run_method(dist, costs, cfg, reps=reps, base_seed=base_seed, store_all=store_all))
    return pd.DataFrame(rows)


def fmt_av_min_max(df_stats: pd.DataFrame) -> pd.Series:
    return df_stats.apply(lambda r: f"{r['avg']:.2f} ({r['min']} – {r['max']})", axis=1)


# -------------------------
# Plots
# -------------------------
def boxplot_methods(df_stats: pd.DataFrame, title: str):
    if "objs" not in df_stats.columns:
        raise ValueError("Need store_all=True so df_stats has an 'objs' column.")

    labels = df_stats["Method"].tolist()
    data = df_stats["objs"].tolist()

    plt.figure(figsize=(14, 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Objective")
    plt.title(title)
    plt.tight_layout()
    plt.show()


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

    # draw the cycle
    path_xy = coords[tour]
    closed = np.vstack([path_xy, path_xy[0]])
    plt.plot(closed[:, 0], closed[:, 1], linewidth=1.2, label="Cycle")

    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def best_solution_overall(df_stats: pd.DataFrame):
    # pick the method/run with the minimal objective
    idx = df_stats["best_obj"].astype(int).idxmin()
    row = df_stats.loc[idx]
    return row["Method"], int(row["best_obj"]), row["best_tour"], row["best_seed_info"]

def to_reference_4row_table(df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 8 rows (Steepest/Greedy x swap/2opt x start) into 4 reference rows:
      - Random starting solution + node exchange
      - Random starting solution + edges exchange
      - Greedy cycle weighted 2-regret starting solution + node exchange
      - Greedy cycle weighted 2-regret starting solution + edges exchange

    We aggregate using the raw 200-run lists in 'objs' (so each row becomes 400 values).
    """
    if "objs" not in df_stats.columns:
        raise ValueError("df_stats must contain 'objs' column (run_instance(..., store_all=True)).")

    def group_label(method_str: str):
        start = (
            "Random starting solution"
            if "Start: random" in method_str
            else "Greedy cycle weighted 2-regret starting solution"
        )
        move = (
            "With node exchange"
            if "Intra: node-swap" in method_str
            else "With edges exchange"
        )
        return start, move

    tmp = df_stats.copy()
    tmp[["Start", "Move"]] = tmp["Method"].apply(lambda s: pd.Series(group_label(s)))

    rows = []
    for (st, mv), g in tmp.groupby(["Start", "Move"]):
        all_vals = []
        for lst in g["objs"]:
            all_vals.extend(lst)  # 200 (greedy) + 200 (steepest) = 400 values
        all_vals = np.asarray(all_vals, dtype=int)

        rows.append({
            "Method": f"{st}\n{mv}",
            "avg": float(all_vals.mean()),
            "min": int(all_vals.min()),
            "max": int(all_vals.max()),
        })

    out = pd.DataFrame(rows)

    # order rows like in your reference image
    order = [
        "Random starting solution\nWith node exchange",
        "Random starting solution\nWith edges exchange",
        "Greedy cycle weighted 2-regret starting solution\nWith node exchange",
        "Greedy cycle weighted 2-regret starting solution\nWith edges exchange",
    ]
    out["__order__"] = out["Method"].map({k:i for i,k in enumerate(order)})
    out = out.sort_values("__order__").drop(columns="__order__").reset_index(drop=True)

    return out


# -------------------------
# Main
# -------------------------
def main():
    instances = [
        ("Instance 1", "TSPA.csv"),
        ("Instance 2", "TSPB.csv"),
    ]

    per_instance = {}
    for name, path in instances:
        coords, dist, costs = read_instance_csv_with_coords(path)
        df_stats = run_instance(coords, dist, costs, reps=200, base_seed=123, store_all=True)
        # stable ordering
        df_stats = df_stats.sort_values("Method").reset_index(drop=True)
        per_instance[name] = (coords, dist, costs, df_stats)

    # ---- Summary table ----
    # table = pd.DataFrame({"Method": per_instance["Instance 1"][3]["Method"]})
    # for inst_name in ["Instance 1", "Instance 2"]:
    #     table[inst_name] = fmt_av_min_max(per_instance[inst_name][3])

    # print("\n=== Summary table (avg (min – max)) ===")
    # print(table.to_string(index=False))
    # ---- Reference summary table (4 rows) ----
    refA = to_reference_4row_table(per_instance["Instance 1"][3])
    refB = to_reference_4row_table(per_instance["Instance 2"][3])

    table = pd.DataFrame({
        "Method": refA["Method"],
        "Instance 1": refA.apply(lambda r: f"{r['avg']:.2f} ({r['min']} – {r['max']})", axis=1),
        "Instance 2": refB.apply(lambda r: f"{r['avg']:.2f} ({r['min']} – {r['max']})", axis=1),
    })

    print("\n=== Reference summary table (avg (min – max)) ===")
    print(table.to_string(index=False))



    # ---- Best tour plot per instance ----
    for inst_name in ["Instance 1", "Instance 2"]:
        coords, dist, costs, df_stats = per_instance[inst_name]
        best_method, best_obj, best_tour, seed_info = best_solution_overall(df_stats)

        length, cs, total = obj_parts(dist, costs, best_tour)

        title = (
            f"{inst_name} best overall\n"
            f"{best_method}\n"
            f"obj={total} (len={length}, cost={cs}), seed_info={seed_info}"
        )
        plot_solution(coords, best_tour, title=title)


if __name__ == "__main__":
    main()



# import numpy as np
# import pandas as pd
# import math

# def round_half_up(x: float) -> int:
#     return int(math.floor(x + 0.5))

# def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
#     n = coords.shape[0]
#     dist = np.zeros((n, n), dtype=np.int32)
#     for i in range(n):
#         xi, yi = coords[i]
#         for j in range(i + 1, n):
#             xj, yj = coords[j]
#             d = math.hypot(int(xi) - int(xj), int(yi) - int(yj))
#             dij = round_half_up(d)
#             dist[i, j] = dij
#             dist[j, i] = dij
#     return dist

# def read_instance_csv(path: str, sep=";"):
#     """
#     CSV format: x ; y ; cost   (no header)
#     returns: dist (int32, n x n), costs (int32, n)
#     """
#     df = pd.read_csv(path, sep=sep, header=None)
#     coords = df[[0, 1]].to_numpy(dtype=np.int32)
#     costs  = df[2].to_numpy(dtype=np.int32)
#     dist = build_distance_matrix(coords)
#     return dist, costs

# import ec_ls

# def run_all_methods_cpp(dist, costs, reps=200, base_seed=123):
#     n = len(costs)
#     m = (n + 1) // 2

#     # search_type: 0 steepest, 1 greedy
#     # intra_type:  0 swap, 1 two_opt
#     # start_type:  0 random, 1 greedy_regret
#     methods = []
#     for search_type in [0, 1]:
#         for intra_type in [0, 1]:
#             for start_type in [0, 1]:
#                 methods.append((search_type, intra_type, start_type))

#     rows = []
#     for (search_type, intra_type, start_type) in methods:
#         objs = []

#         if start_type == 0:
#             # 200 random starts
#             for r in range(reps):
#                 _, obj, _ = ec_ls.local_search(
#                     dist, costs, m,
#                     search_type, intra_type, start_type,
#                     base_seed + r, -1
#                 )
#                 objs.append(int(obj))
#         else:
#             # use each of the 200 nodes as starting node (or min(reps,n))
#             for start_node in range(min(reps, n)):
#                 _, obj, _ = ec_ls.local_search(
#                     dist, costs, m,
#                     search_type, intra_type, start_type,
#                     base_seed + start_node, start_node
#                 )
#                 objs.append(int(obj))

#         avg = float(np.mean(objs))
#         mn  = int(np.min(objs))
#         mx  = int(np.max(objs))

#         s = "Steepest" if search_type == 0 else "Greedy"
#         intra = "Intra: node-swap" if intra_type == 0 else "Intra: 2-opt"
#         st = "Start: random" if start_type == 0 else "Start: greedy-regret"

#         rows.append({
#             "Method": f"{s} | {intra} | {st}",
#             "avg": avg,
#             "min": mn,
#             "max": mx,
#         })

#     return pd.DataFrame(rows).sort_values("Method").reset_index(drop=True)

# def format_av_min_max(df_stats: pd.DataFrame, col_name: str):
#     return df_stats.apply(lambda r: f"{r['avg']:.2f} ({r['min']} – {r['max']})", axis=1).rename(col_name)

# # Read both instances
# distA, costsA = read_instance_csv("TSPA.csv")
# distB, costsB = read_instance_csv("TSPB.csv")

# # Run experiments
# statsA = run_all_methods_cpp(distA, costsA, reps=200, base_seed=123)
# statsB = run_all_methods_cpp(distB, costsB, reps=200, base_seed=123)

# # Merge into one table (like your screenshot)
# table = pd.DataFrame({
#     "Method": statsA["Method"],
#     "Instance 1": format_av_min_max(statsA, "Instance 1"),
#     "Instance 2": format_av_min_max(statsB, "Instance 2"),
# })

# table.head()
# print(table)

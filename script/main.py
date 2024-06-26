# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
import warnings
from time import time, perf_counter
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import power_grid_model as pgm
from grid_gen import generate_fictional_grid
from power_grid_model.validation import errors_to_string, validate_batch_data, validate_input_data
from tensorpowerflow import GridTensor
from tqdm import tqdm

warnings.filterwarnings("ignore")
import platform
from datetime import datetime


class power_flow:
    pgm = {
        "newton_raphson": "PGM Newton-Raphson",
        "linear": "PGM Linear Impedance",
        "linear_current": "PGM Linear Current",
        "iterative_current": "PGM Iterative Current",
    }
    tpf = {
        "tensor": "TPF Tensor",
        "hp-tensor": "TPF Tensor-Sparse",
    }
    pgm_algorithms = list(product(["pgm"], pgm.values()))
    tpf_algorithms = list(product(["tpf"], tpf.values()))


def experiment(n_feeder=20, n_node_per_feeder=50, n_step=1_000, log=False):
    def log_print(*args):
        if log:
            print(*args)

    # fictional grid parameters
    cable_length_km_min = 0.8
    cable_length_km_max = 1.2
    # load_p_w_min = 0.4e6 * 0.8
    # load_p_w_max = 0.4e6 * 1.2
    load_p_w_min = 0.4e6 * 0.001
    load_p_w_max = 0.4e6 * 0.001
    pf = 0.95

    load_scaling_min = 0.5
    load_scaling_max = 1.5

    # gen grid data
    fictional_dataset = generate_fictional_grid(
        n_node_per_feeder=n_node_per_feeder,
        n_feeder=n_feeder,
        cable_length_km_min=cable_length_km_min,
        cable_length_km_max=cable_length_km_max,
        load_p_w_max=load_p_w_max,
        load_p_w_min=load_p_w_min,
        pf=pf,
        n_step=n_step,
        load_scaling_min=load_scaling_min,
        load_scaling_max=load_scaling_max,
    )
    # unpack data
    pgm_dataset = fictional_dataset["pgm_dataset"]
    pgm_update_dataset = fictional_dataset["pgm_update_dataset"]
    tpf_node_data = fictional_dataset["tpf_grid_nodes"]
    tpf_line_data = fictional_dataset["tpf_grid_lines"]
    tpf_time_series_p = fictional_dataset["tpf_time_series_p"]
    tpf_time_series_q = fictional_dataset["tpf_time_series_q"]

    # validate data
    log_print(errors_to_string(validate_input_data(pgm_dataset)))
    log_print(errors_to_string(validate_batch_data(pgm_dataset, pgm_update_dataset)))

    res_pgm = []
    # create grids, run pf's and time them
    # pgm - all 4 methods

    for pgm_algorithm in power_flow.pgm.keys():
        pgm_start_time = time()
        model_instance = pgm.PowerGridModel(pgm_dataset)
        start = perf_counter()
        result_pgm = model_instance.calculate_power_flow(
            symmetric=True,
            calculation_method=pgm_algorithm,
            update_data=pgm_update_dataset,
            output_component_types=["node", "line"],
            max_iterations=10000,
        )

        end = perf_counter()
        pgm_end_time = time()
        res_pgm.append([end - start])
        log_print(f"{power_flow.pgm[pgm_algorithm]}: {end - start}")
        log_print(f"Total time{power_flow.pgm[pgm_algorithm]}: {pgm_end_time - pgm_start_time}")

        if pgm_algorithm == "newton_raphson":
            voltage_results_pgm = result_pgm["node"]["u_pu"][0, 1:]  # First node is slack.

    # tpf
    # tpf_time_start = time()
    tpf_instance = GridTensor(
        node_file_path="",
        lines_file_path="",
        from_file=False,
        nodes_frame=tpf_node_data,
        lines_frame=tpf_line_data,
        gpu_mode=False,
        numba=True,
        v_base=10.0,  # kV
    )
    # tpf_time_end = time()

    res_tpf = []
    for tpf_algorithm in power_flow.tpf.keys():
        start = time()
        result = tpf_instance.run_pf(
            active_power=tpf_time_series_p,
            reactive_power=tpf_time_series_q,
            algorithm=tpf_algorithm,
            tolerance=1e-6,
            sparse_solver="pardiso",
        )
        assert result["convergence"], "TPF did not converge. Check results."
        end = time()
        res_tpf.append(result["time_pf"])
        log_print(f"TensorPowerFlow.{tpf_algorithm}: {end - start}")

        assert np.allclose(voltage_results_pgm, np.abs(result["v"][0, :])), "PGM and TensorPF does not match."

    pgm_results = dict(zip(power_flow.pgm_algorithms, res_pgm))
    tpf_results = dict(zip(power_flow.tpf_algorithms, res_tpf))

    return pgm_results | tpf_results


if __name__ == "__main__":
    # exp_options = [[n_feeder, n_node_per_feeder]]
    # N_STEPS = [10, 100, 1_000, 10_000, 50_000]
    N_STEPS = [10, 100, 1_000, 5_000]
    # exp_options = [[20, 1], [20, 5], [20, 10], [20, 20], [20, 25], [30, 20], [20, 50]]
    exp_options = [[20, 1], [20, 5], [20, 10], [20, 20], [20, 25], [30, 20], [20, 50], [40, 50], [60, 50], [80, 50], [100, 50], [100,60]]

    results_exp = None
    for n_steps in tqdm(N_STEPS):
        print(f"N_STEPS: {n_steps}")
        result_nodes = None
        for option in exp_options:
            print(f"Nodes: {option[0]*option[1]}")
            res = experiment(option[0], option[1], n_step=n_steps)
            result_frame = pd.DataFrame(res | {("nodes", ""): option[0] * option[1], ("n_steps", ""): n_steps})
            result_nodes = pd.concat([result_nodes, result_frame], axis=0, ignore_index=True)
        results_exp = pd.concat([results_exp, result_nodes], axis=0, ignore_index=True)

    results_exp.to_pickle("../data/results_windows.pkl")

    # %%
    results_exp = pd.read_pickle("../data/results_windows.pkl")
    steps = results_exp["n_steps"].unique()
    n_steps = len(steps)

    fig, ax = plt.subplots(1, len(results_exp["n_steps"].unique()), figsize=(20, 3.5))
    plt.subplots_adjust(wspace=0.3, right=0.95, top=0.90, bottom=0.15)

    line_styles = ["-", "--", ":", "-."]
    for ii, n_steps in tqdm(enumerate(steps)):
        pgm_color = 0
        tpf_color = 0
        exp_results = results_exp[results_exp["n_steps"] == n_steps].drop("n_steps", axis=1)
        exp_results.set_index("nodes", inplace=True)
        for algo, result in exp_results.T.iterrows():
            print(algo)
            if algo[0] == "pgm":
                color = plt.cm.tab20c(pgm_color)
                line_style = line_styles[pgm_color]
                pgm_color += 1
            else:
                color = plt.cm.tab20c(tpf_color + 4)
                line_style = line_styles[tpf_color]
                tpf_color += 1
            ax[ii].semilogy(
                result.index, result.values, linewidth=1.8, marker=".", label=algo[1], linestyle=line_style, color=color
            )
            ax[ii].set_title(f"Power flows: {n_steps}")

    for ax_ in ax:
        ax_.legend(fontsize="xx-small", handlelength=2.5)
        ax_.grid()
        ax_.set_xlabel("Number of Nodes")
        ax_.set_ylabel("Time (s)")

    plt.savefig(f"../data/benchmark_{platform.system()}_{datetime.now().strftime('%a%H%M%S')}.pdf")
    plt.show()

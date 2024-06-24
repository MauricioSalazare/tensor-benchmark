# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from matplotlib import pyplot as plt


class BenchmarkPlotter:
    def __init__(self):
        self.data_pgm_li = []
        self.data_pgm_lc = []
        self.data_pgm_ic = []
        self.data_pgm_nr = []
        self.data_tpf_pf = []
        self.data_tpf_tf = []
        self.data_tpf_tf_hl = []
        self.n_nodes = []

    def add(self, res, n_nodes):
        res_pgm = res["result pgm"]
        res_tpf = res["result tpf"]
        self.data_pgm_li.append(res_pgm[0])
        self.data_pgm_lc.append(res_pgm[1])
        self.data_pgm_ic.append(res_pgm[2])
        self.data_pgm_nr.append(res_pgm[3])
        self.data_tpf_pf.append(res_tpf[0])
        self.data_tpf_tf.append(res_tpf[1])
        self.data_tpf_tf_hl.append(res_tpf[2])
        self.n_nodes.append(n_nodes)

    def plot(self, log_scale=False):
        plt.figure(figsize=(8, 5))
        _, ax = plt.subplots()
        data_lists = [
            self.data_pgm_li,
            self.data_pgm_lc,
            self.data_pgm_ic,
            self.data_pgm_nr,
            self.data_tpf_pf,
            self.data_tpf_tf,
            self.data_tpf_tf_hl,
        ]
        labels = ["pgm li", "pgm lc", "pgm ic", "pgm nr", "tpf pf", "tpf tf", "tpf tf_hl"]

        for data_list, label in zip(data_lists, labels):
            if log_scale:
                ax.semilogy(self.n_nodes, data_list, label=label)
            else:
                ax.plot(self.n_nodes, data_list, label=label)

        ax.legend()
        plt.show()

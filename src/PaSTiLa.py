"""
Project: PasTiLa (Parallel Automatic Snippet-based Time series Labeling Algorithm)
Source file: PasTiLa.cu
Purpose: Parallel algrorithm for labeling long time series
Author(s): Andrey Goglachev (goglachevai@susu.ru)
"""

import numpy as np
import cupy as cp
from mpi4py import MPI
from sklearn.preprocessing import PolynomialFeatures
from numberpartitioning import karmarkar_karp
import pickle


def pastila(input_filename, min_m, max_m, K, step=1, output_filename="output.pkl"):
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    with open("PaTILa.h", "r") as f:
        raw = f.read()

    module = cp.RawModule(code=raw)
    pastila_kernel = module.get_function("pastila")

    if rank == 0:
        with open("predictor.pkl", "rb") as f:
            predictor = pickle.load(f)

        ts = np.loadtxt(input_filename)
        comm.Bcast([ts, MPI.DOUBLE], root=0)
        n = len(ts)
        ts_gpu = cp.array(ts)

        m_range = np.arange(min_m, max_m, step)
        runtimes = np.empty_like(m_range)
        for i, m in enumerate(m_range):
            runtimes[i] = predictor.predict((len(ts), m))
        parts = karmarkar_karp(runtimes, num_parts=nprocs).indices

        for i in range(1, nprocs):
            m_to_proccess = [m_range[index] for index in parts[i]]
            req = comm.isend(m_to_proccess, dest=i, tag=1)
            req.wait()

        m_to_proccess = [m_range[index] for index in parts[0]]

        snippets = []
        for m in m_to_proccess:
            snp = pastila_kernel(ts_gpu, n, m, K)
            snippets.append(snp)

        gathered_snippets = comm.gather(cp.asnumpy(snippets), root=0)

        max_area = -np.inf
        for snp in gathered_snippets:
            profiles = snp[1]
            max_dist = np.max(profiles)
            total_area = 0
            for i, p_1 in enumerate(profiles[:-1]):
                for p_2 in profiles[i:]:
                    total_area += sum(np.abs(p_1 - p_2))
            if total_area > max_area:
                best_snp = snp

        if output_filename is not None:
            with open(output_filename, "wb") as f:
                pickle.dump(best_snp, f)
        else:
            return best_snp

    else:
        ts = comm.Bcast([ts, MPI.DOUBLE], root=0)
        ts_gpu = cp.array(ts)
        n = len(ts)
        req = comm.irecv(source=0, tag=1)
        m_to_proccess = data = req.wait()

        snippets = []
        for m in m_to_proccess:
            snp = pastila_kernel(ts_gpu, n, m, K)
            snippets.append(snp)

        comm.gather(cp.asnumpy(snippets), root=0)

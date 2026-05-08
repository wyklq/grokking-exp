#%% 
import pandas as pd
import matplotlib.pyplot as plt

cells = pd.read_parquet("../results/scans/C_p113_phase1.cells.parquet")
traj  = pd.read_parquet("../results/scans/C_p113_phase1.parquet")

# 相图：每个 (alpha_idx, lambda_idx) 的 phase
phase_map = cells.pivot(index="lambda_idx", columns="alpha_idx", values="phase")
print(phase_map)

# 单 cell 的 fourier_sparsity 轨迹
cell = traj[(traj.alpha_idx == 4) & (traj.lambda_idx == 3)]
plt.semilogx(cell.step, cell.fourier_sparsity)
plt.xlabel("step"); plt.ylabel("Fourier sparsity (Gini)")

# (Track1, Track2) 解释矩阵：每 cell 取最终一步
last = traj.sort_values("step").drop_duplicates(["alpha_idx","lambda_idx","seed"], keep="last")
last = last.merge(cells[["alpha_idx","lambda_idx","phase"]].drop_duplicates(),
                  on=["alpha_idx","lambda_idx"], how="left")
plt.scatter(last.fourier_sparsity, last.weight_norm_total,
            c=last.phase.astype("category").cat.codes)

# %%

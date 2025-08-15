# 晶創主機

## 常用流程

- 查詢可用節點: `sinfo --states=idle`
- 修改要使用的 partition 和 task,cpu,gpu in `job.sh`
  - 記得修改 `email` 欄位
- 排程送件: `sbatch job.sh`
- 查看目前的狀態:
  - 查看所有人的資訊: `squeue -l`
  - 查看自己的排程資訊: `squeue -u <username>`
- ssh 進去執行的節點: ex. `ssh hgpn02` (依據上一指令決定)
  - `ml load miniconda3`
  - `conda activate videograin`
  - `pip install nvitop`
  - `nvitop`
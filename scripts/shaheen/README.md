# 在 Shaheen III 上跑 HermesSim 的 lift 和 SOG 预处理

流程分四步：CFG summary → gsat P-Code lift（多节点）→ SOG/ISCG/TSCG 预处理（单节点）→ 训练和推理（GPU，不在本目录）。

## 0. 一次性准备（login 节点，计算节点不能上网）

```bash
git clone -b cluster https://github.com/damaoooo/HermesSim.git /scratch/$USER/HermesSim
cd /scratch/$USER/HermesSim
bash scripts/shaheen/setup.sh     # uv 环境 .venv、私有 JDK 21 (.jdk/)、bin/gsat-1.0-opt.jar
mkdir -p logs
```

`gsat/build.sh` 会先下载 gsat release jar v1.0.0 并校验 sha256，再把 `gsat/src` 里的 com.gsat 类重新编译进去。release jar 里带的是 HermesSim 修改过的 Ghidra 10.3。

## 1. CFG summary

每个二进制生成一个 `<bin>_cfg_summary.json`，内容和 `lifting/dataset_summary.py` 逐字节相同。输入是 Talos CSV 和 IDA 的 `acfg_features`。

建议在数据所在的机器上生成 summary，只把结果（几 GB）传上来；`acfg_features` 有 29 GB，没必要传。

```bash
for s in training validation testing; do
  .venv/bin/python lifting/lift_cluster.py summaries --workers 16 \
    --dataset_info_csv <protocol>/${s}_Dataset-1.csv \
    --cfgs_folder <features>/${s}/acfg_features_Dataset-1_${s} \
    --cfg_summary dbs/Dataset-1/cfg_summary/${s}
done
```

## 2. lift（多节点 array）

`<binaries_root>` 下面可以是 `Dataset-1/<project>/<bin>`（`idb_path` 去掉开头的 `IDBs/` 和结尾的 `.i64`），也可以是 ReGraphv2 那种按 split 分开的 `Dataset-1/{train,validation,test}/<project>/<bin>`，两种都会自动识别。

```bash
for s in training validation testing; do
  .venv/bin/python lifting/lift_cluster.py plan \
    --cfg_summary dbs/Dataset-1/cfg_summary/$s --binaries_root <binaries_root> \
    --output_dir /scratch/$USER/hermes/lift/$s --jobs /scratch/$USER/hermes/jobs_$s.jsonl
done
sbatch --array=0-7 scripts/shaheen/lift.sbatch \
    /scratch/$USER/hermes/jobs_testing.jsonl /scratch/$USER/hermes/lift/testing
.venv/bin/python lifting/lift_cluster.py check \
    --jobs /scratch/$USER/hermes/jobs_testing.jsonl --output_dir /scratch/$USER/hermes/lift/testing
```

- `plan` 写出的 jobs 文件里是绝对路径，所以必须在 Shaheen 上运行。
- 输出文件是 `<bin>_acfg_disasm.json.gz`，由 gsat 原子写入。中断后用同一条命令重新提交即可续跑，已完成的会自动跳过。
- 每个 array task 处理 jobs 的一个分片，分片按估计代价做 LPT 均衡。节点内的调度：
  - 大二进制先跑，同时受 CPU 和内存两个预算约束；
  - 小二进制按批放进同一个 JVM（gsat 的 `batch` 模式），省掉重复的启动开销；
  - 失败的任务单独重试，每次堆内存翻倍。
- 每个二进制的结果记录在 `_status/shard_*.jsonl`，gsat 日志在 `_logs/`。

建议参数：

- `THREADS=1`（默认）：单个 JVM 多线程几乎不提速。z3 x64（29 MB）用 1、3、32 线程分别是 508、494、478 秒，单线程内存最少（3.1 GB 对 7.6 GB）。单线程结果也可复现：在两台机器、JDK 19 和 21 上完全一致。多线程在机器繁忙时偶尔不一致：一次 32 线程的运行在 2,681 个函数中有 3 个的 callee 参数个数不同。
- `ANALYSIS_TIMEOUT`：默认 14400 秒。gsat 原版默认 1200 秒，到时间就取消自动分析，然后用残缺的分析结果继续 lift，这会让结果取决于机器速度。单线程下最大的 z3（104 MB）会超过 1200 秒，所以默认放宽。`check` 会列出仍然触发了超时的二进制。
- `DISABLE_ANALYZERS`：默认为空。实测只有 `Function ID` 关掉后输出不变，约快 7%，但目前只验证了 8 个二进制。其他耗时的 analyzer（Call Convention ID、Demangler GNU、GCC Exception Handlers）关掉都会改变 SOG。

## 3. 预处理（单节点）

```bash
sbatch scripts/shaheen/preprocess.sbatch /scratch/$USER/hermes/lift /scratch/$USER/hermes/inputs/pcode
```

`<lift_root>/{training,validation,testing}` 是三个 split 的 lift 输出。训练集负责建词表，验证集和测试集复用这份词表。

`.json` 和 `.json.gz` 都能读。词表在计数相同的 token 之间按字符串排序，所以结果不再依赖文件列举顺序或 worker 完成顺序；原版在这一点上不确定。

跑完后，`preprocess/update_vocab_config.py` 会把新词表的大小写进 `model/configures/` 里的 e00/e02 配置：SOG、TSCG 写入 `n_node_attr`，ISCG 写入 gru 的 `embed_size`。如果之后在别处重新预处理，也要重新运行这一步。

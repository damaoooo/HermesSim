#!/usr/bin/env python3
##############################################################################
# Scalable P-Code lifting for HermesSim (replaces dataset_summary.py + pcode_lifter.py for large runs).
#
#   summaries  CSV + IDA acfg_features -> one <bin>_cfg_summary.json per binary (same content as
#              lifting/dataset_summary.py), built in parallel, resumable, errors reported.
#   plan       cfg summaries -> jobs.jsonl (binary, size, #functions, output path), largest first.
#   run        Lift the jobs of one shard on one node with gsat's batch mode: several JVMs, each
#              processing a chunk of binaries so that Ghidra start-up and JIT warm-up are paid once per
#              chunk. Largest-first scheduling under a CPU and a memory budget; outputs are written
#              atomically by gsat, finished outputs are skipped, failed jobs are retried alone with a
#              doubled heap.
#   check      Report missing outputs of a jobs file and their last status.
#
# The lifting itself is unchanged: gsat pcode-extractor-v2 -m elf -of ALL -v 1 -opt 0 (the HermesSim
# setting), written gzip-compressed as <bin>_acfg_disasm.json.gz unless --no-gzip is given to plan.
##############################################################################

import argparse
import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import basename, join

CFG_SUMMARY_POSTFIX = "_cfg_summary.json"
ACFG_POSTFIX = "_acfg_disasm.json"


# ---------------------------------------------------------------------------- summaries

def _bin_name(idb_path):
    assert idb_path.endswith(".i64") or idb_path.endswith(".idb"), idb_path
    return basename(idb_path)[:-4]


def _write_summary(args):
    """dataset_summary.py's output for one binary."""
    idb_path, funcs, cfgs_folder, out_dir, force = args
    bin_name = _bin_name(idb_path)
    out_fp = join(out_dir, bin_name + CFG_SUMMARY_POSTFIX)
    if not force and os.path.exists(out_fp):
        return idb_path, "skipped", None
    try:
        with open(join(cfgs_folder, bin_name + "_acfg_features.json")) as f:
            acfg_feats = json.load(f)[idb_path]
        finfos = []
        for start_ea, func_name in funcs:
            fa = acfg_feats[start_ea]
            finfos.append({
                "start_ea": start_ea,
                "func_name": func_name,
                "nodes": [[hex(int(bb_va)), bb["bb_len"]] for bb_va, bb in fa["basic_blocks"].items()],
                "edges": fa["edges"],
            })
        tmp = out_fp + ".tmp"
        with open(tmp, "w") as f:
            json.dump({idb_path: finfos}, f)
        os.replace(tmp, out_fp)
        return idb_path, "ok", None
    except Exception as e:  # noqa: BLE001 - reported by the caller
        return idb_path, "error", repr(e)


def cmd_summaries(a):
    import pandas as pd

    df = pd.read_csv(a.dataset_info_csv, usecols=["idb_path", "fva", "start_ea", "func_name"], dtype=str)
    bad = df.fva != df.start_ea
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} rows have fva != start_ea (dataset_summary.py asserts this)")
    os.makedirs(a.cfg_summary, exist_ok=True)
    groups = defaultdict(list)  # functions in CSV row order, as dataset_summary.py
    for idb_path, start_ea, func_name in zip(df.idb_path, df.start_ea, df.func_name):
        groups[idb_path].append((start_ea, func_name))
    tasks = sorted(((k, v, a.cfgs_folder, a.cfg_summary, a.force) for k, v in groups.items()), key=lambda t: -len(t[1]))
    counts, errors = defaultdict(int), []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_write_summary, t) for t in tasks]), 1):
            idb_path, status, err = fut.result()
            counts[status] += 1
            if err:
                errors.append(idb_path)
                print(f"[!] {idb_path}: {err}", file=sys.stderr)
            if i % 200 == 0 or i == len(tasks):
                print(f"[summaries] {i}/{len(tasks)} {dict(counts)}", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} binaries failed")


# ---------------------------------------------------------------------------- plan

def idb_path_to_binary_path(idb_path, binaries_root):
    """pcode_lifter.py's mapping IDBs/<rel>.i64 -> binaries/<rel>, with a configurable root."""
    assert idb_path.startswith("IDBs/"), idb_path
    return join(binaries_root, idb_path[len("IDBs/"):-4])


def cmd_plan(a):
    jobs, missing = [], []
    for fn in sorted(os.listdir(a.cfg_summary)):
        if not fn.endswith(CFG_SUMMARY_POSTFIX):
            continue
        fp = join(a.cfg_summary, fn)
        with open(fp) as f:
            idb_path, funcs = next(iter(json.load(f).items()))
        bin_fp = idb_path_to_binary_path(idb_path, a.binaries_root)
        if not os.path.exists(bin_fp):
            missing.append(bin_fp)
            continue
        out_name = basename(bin_fp) + ACFG_POSTFIX + (".gz" if a.gzip else "")
        jobs.append({"idb_path": idb_path, "binary": os.path.abspath(bin_fp), "cfg_summary": os.path.abspath(fp),
                     "output": os.path.abspath(join(a.output_dir, out_name)),
                     "bytes": os.path.getsize(bin_fp), "functions": len(funcs)})
    if missing:
        print(f"[!] {len(missing)} binaries not found, e.g. {missing[:3]}", file=sys.stderr)
        if not a.allow_missing:
            raise SystemExit(1)
    jobs.sort(key=lambda j: (-j["bytes"], j["binary"]))
    with open(a.jobs, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"[plan] {len(jobs)} jobs, {sum(j['bytes'] for j in jobs) / 2**30:.1f} GiB of binaries, "
          f"{sum(j['functions'] for j in jobs)} functions -> {a.jobs}")


def load_jobs(fp):
    with open(fp) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------- run

def job_cost(j):
    """Relative run time: auto-analysis dominates and grows with the binary; plus a fixed per-binary overhead."""
    return j["bytes"] / 2**20 + 3.0


def shard_jobs(jobs, shard, num_shards):
    """Deterministic largest-first (LPT) assignment of jobs to shards by estimated cost."""
    loads, mine = [0.0] * num_shards, []
    for j in sorted(jobs, key=lambda j: (-job_cost(j), j["binary"])):
        s = min(range(num_shards), key=lambda i: (loads[i], i))
        loads[s] += job_cost(j)
        if s == shard:
            mine.append(j)
    return mine


def est_mem_gb(j):
    """Resident memory of one gsat JVM (measured: 8 MiB nmap -> 2.9 GB, 29 MiB z3 -> 5.7 GB)."""
    return 2.0 + 0.15 * j["bytes"] / 2**20


def make_chunks(jobs, max_chunk_mb, max_chunk_jobs, big_mb):
    """Big binaries run alone; small ones are grouped, similar sizes together, to amortize JVM start-up."""
    chunks, cur, cur_mb = [], [], 0.0
    for j in sorted(jobs, key=lambda j: (-j["bytes"], j["binary"])):
        mb = j["bytes"] / 2**20
        if mb >= big_mb:
            chunks.append([j])
            continue
        if cur and (cur_mb + mb > max_chunk_mb or len(cur) >= max_chunk_jobs):
            chunks.append(cur)
            cur, cur_mb = [], 0.0
        cur.append(j)
        cur_mb += mb
    if cur:
        chunks.append(cur)
    return chunks


class Runner:
    def __init__(self, a, jobs):
        self.a = a
        self.cv = threading.Condition()
        self.free_cpu, self.free_mem = a.cores, a.mem_gb
        self.queue = make_chunks(jobs, a.chunk_mb, a.chunk_jobs, a.big_mb)
        self.attempts = defaultdict(int)
        self.counts = defaultdict(int)
        self.running = 0
        self.next_id = 0
        self.host = socket.gethostname()
        self.status_fp = join(a.output_dir, "_status", f"shard_{a.shard:04d}.jsonl")
        self.log_dir = join(a.output_dir, "_logs")
        os.makedirs(os.path.dirname(self.status_fp), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.local = tempfile.mkdtemp(prefix="hermes-lift-", dir=a.local_tmp)

    def record(self, job, status, secs, attempt, heap, error=None, analysis_timeout=False):
        rec = {"binary": job["binary"], "output": job["output"], "status": status, "secs": round(secs, 2),
               "attempt": attempt, "heap_gb": heap, "analysis_timeout": analysis_timeout, "host": self.host,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if error:
            rec["error"] = error
        with self.cv:
            self.counts[status] += 1
            with open(self.status_fp, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def heap_gb(self, chunk, attempt):
        need = max(est_mem_gb(j) for j in chunk) * 2
        heap = self.a.heap_gb
        while heap < need:
            heap *= 2
        return min(heap * 2 ** attempt, self.a.max_heap_gb)

    def mem_need(self, chunk, attempt):
        """Memory reserved for a chunk: estimated RSS, more on retries (a retry may follow an OOM)."""
        return min(self.heap_gb(chunk, attempt), max(est_mem_gb(j) for j in chunk) * 1.3 * 2 ** attempt + 1)

    def run_chunk(self, chunk, attempt, cid):
        heap = self.heap_gb(chunk, attempt)
        cdir = join(self.local, f"chunk{cid}")
        home, tmp = join(cdir, "home"), join(cdir, "tmp")
        os.makedirs(home)
        os.makedirs(tmp)
        jobs_tsv, status_jsonl = join(cdir, "jobs.tsv"), join(cdir, "status.jsonl")
        with open(jobs_tsv, "w") as f:
            for j in chunk:
                args = ["pcode-extractor-v2", "-m", "elf", "-f", j["binary"], "-c", j["cfg_summary"],
                        "-of", self.a.graph_type, "-v", "1", "-opt", "0", "-at", str(self.a.analysis_timeout)]
                if self.a.disable_analyzers:
                    args += ["-da", self.a.disable_analyzers]
                f.write("\t".join(args + ["-o", j["output"]]) + "\n")
        # Ghidra keeps its log, preferences, temporary projects and caches under user.home and java.io.tmpdir;
        # keep them node-local and private to this JVM (no shared files on Lustre).
        cmd = ["nice", "-n", "5", self.a.java, "-ea", f"-Xmx{heap}G", "-XX:+UseCompressedOops",
               f"-XX:ActiveProcessorCount={self.a.threads}", f"-Duser.home={home}", f"-Djava.io.tmpdir={tmp}",
               "-jar", self.a.jar, "batch", jobs_tsv, status_jsonl]
        log_fp = join(self.log_dir, f"s{self.a.shard:04d}_c{cid:05d}_a{attempt}_{basename(chunk[0]['binary'])}.log.gz")
        t0 = time.time()
        timed_out, cur = set(), None  # gsat cancels auto-analysis at the timeout and lifts the partial result
        with gzip.open(log_fp, "wt") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
            for line in proc.stdout:
                log.write(line)
                if "[BATCH] job " in line:
                    cur = int(line.split("[BATCH] job ")[1].split("/")[0]) - 1
                elif "Analysis Timeout" in line and cur is not None:
                    timed_out.add(cur)
            code = proc.wait()
        done = {}
        if os.path.exists(status_jsonl):
            with open(status_jsonl) as f:
                for line in f:
                    r = json.loads(line)
                    done[r["job"]] = r
        failed = []
        for i, j in enumerate(chunk):
            r = done.get(i)
            ok = r is not None and r["status"] in ("ok", "skipped") and os.path.exists(j["output"])
            status = r["status"] if r else f"jvm_exit_{code}"
            self.record(j, status if ok or status != "ok" else "missing_output",
                        (r or {}).get("secs", time.time() - t0), attempt, heap, None if ok else (r or {}).get("error"),
                        i in timed_out)
            if not ok:
                failed.append(j)
        shutil.rmtree(cdir, ignore_errors=True)
        return failed

    def worker(self, chunk, attempt, cid, cpu, mem):
        failed = []
        try:
            failed = self.run_chunk(chunk, attempt, cid)
        except Exception as e:  # noqa: BLE001
            print(f"[!] chunk {cid} driver error: {e!r}", file=sys.stderr, flush=True)
            failed = chunk
        finally:
            with self.cv:
                for j in failed:  # retry alone, with a doubled heap
                    self.attempts[j["output"]] += 1
                    if self.attempts[j["output"]] <= self.a.retries:
                        self.queue.insert(0, [j])
                self.free_cpu += cpu
                self.free_mem += mem
                self.running -= 1
                self.cv.notify_all()

    def run(self):
        print(f"[run] host={self.host} shard={self.a.shard}/{self.a.num_shards} chunks={len(self.queue)} "
              f"cores={self.a.cores} mem={self.a.mem_gb:.0f}G threads/jvm={self.a.threads}", flush=True)
        last = 0.0
        while True:
            with self.cv:
                while True:
                    if time.time() - last > 300:
                        last = time.time()
                        print(f"[run] {time.strftime('%H:%M:%S')} {dict(self.counts)} queued={len(self.queue)} "
                              f"running={self.running}", flush=True)
                    if not self.queue and self.running == 0:
                        shutil.rmtree(self.local, ignore_errors=True)
                        print(f"[run] finished {dict(self.counts)}", flush=True)
                        return
                    if self.queue:
                        chunk = self.queue[0]
                        attempt = max(self.attempts[j["output"]] for j in chunk)
                        cpu, mem = self.a.threads, self.mem_need(chunk, attempt)
                        # Always start something when idle, even if it exceeds the budget on its own.
                        if self.running == 0 or (self.free_cpu >= cpu and self.free_mem >= mem):
                            self.queue.pop(0)
                            self.free_cpu -= cpu
                            self.free_mem -= mem
                            self.running += 1
                            cid, self.next_id = self.next_id, self.next_id + 1
                            break
                    self.cv.wait(timeout=30)
            threading.Thread(target=self.worker, args=(chunk, attempt, cid, cpu, mem), daemon=True).start()


def cmd_run(a):
    jobs = shard_jobs(load_jobs(a.jobs), a.shard, a.num_shards)
    os.makedirs(a.output_dir, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(j["output"])]
    print(f"[run] shard {a.shard}: {len(jobs)} jobs, {len(jobs) - len(todo)} already done", flush=True)
    if todo:
        Runner(a, todo).run()
    left = [j for j in jobs if not os.path.exists(j["output"])]
    if left:
        print(f"[run] {len(left)} outputs still missing, e.g. {[basename(j['binary']) for j in left[:5]]}", flush=True)
    return 1 if left else 0


def cmd_check(a):
    jobs = load_jobs(a.jobs)
    missing = [j for j in jobs if not os.path.exists(j["output"])]
    last = {}
    sdir = join(a.output_dir, "_status")
    if os.path.isdir(sdir):
        for fn in sorted(os.listdir(sdir)):
            with open(join(sdir, fn)) as f:
                for line in f:
                    r = json.loads(line)
                    last[r["output"]] = r
    print(f"[check] {len(jobs) - len(missing)}/{len(jobs)} outputs present")
    tos = [r for r in last.values() if r.get("analysis_timeout")]
    if tos:
        print(f"[check] {len(tos)} outputs were lifted after an auto-analysis timeout (partial analysis), e.g. "
              f"{[basename(r['binary']) for r in tos[:5]]}; rerun them with a larger --analysis-timeout")
    for j in missing[:50]:
        r = last.get(j["output"], {})
        print(f"  missing {basename(j['binary'])}: last status {r.get('status')} {r.get('error', '')}")
    return 1 if missing else 0


def main():
    p = argparse.ArgumentParser(description="Scalable gsat lifting for HermesSim.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summaries", help="Build <bin>_cfg_summary.json files (dataset_summary.py equivalent).")
    s.add_argument("--cfg_summary", required=True, help="Output directory.")
    s.add_argument("--dataset_info_csv", required=True)
    s.add_argument("--cfgs_folder", required=True, help="Folder of <bin>_acfg_features.json files.")
    s.add_argument("--workers", type=int, default=os.cpu_count())
    s.add_argument("--force", action="store_true", help="Rewrite existing summaries.")

    s = sub.add_parser("plan", help="Write the jobs file.")
    s.add_argument("--cfg_summary", required=True)
    s.add_argument("--binaries_root", required=True, help="Directory that replaces 'IDBs' in idb_path.")
    s.add_argument("--output_dir", required=True)
    s.add_argument("--jobs", required=True)
    s.add_argument("--no-gzip", dest="gzip", action="store_false", help="Write plain JSON outputs.")
    s.add_argument("--allow-missing", action="store_true")

    s = sub.add_parser("run", help="Lift one shard of the jobs on this node.")
    s.add_argument("--jobs", required=True)
    s.add_argument("--output_dir", required=True, help="Where _status/ and _logs/ go (outputs follow the jobs file).")
    s.add_argument("--jar", required=True)
    s.add_argument("--java", default="java")
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--cores", type=int, default=os.cpu_count())
    s.add_argument("--mem-gb", type=float, default=None, help="Memory budget (default 90%% of RAM).")
    s.add_argument("--threads", type=int, default=1,
                   help="CPU threads per JVM (-XX:ActiveProcessorCount). z3 (29 MiB): 1 -> 508 s, 3 -> 494 s, 32 -> 478 s.")
    s.add_argument("--heap-gb", type=int, default=16, help="Minimum -Xmx (HermesSim uses 16G).")
    s.add_argument("--max-heap-gb", type=int, default=128)
    s.add_argument("--retries", type=int, default=2)
    s.add_argument("--chunk-mb", type=float, default=40, help="Max MiB of small binaries per JVM.")
    s.add_argument("--chunk-jobs", type=int, default=64)
    s.add_argument("--big-mb", type=float, default=20, help="Binaries of at least this size get their own JVM.")
    s.add_argument("--analysis-timeout", type=int, default=14400,
                   help="gsat auto-analysis timeout (s). gsat cancels the analysis at this point and lifts the partial "
                        "result; its own default of 1200 s is too short for the largest z3 binaries on one thread.")
    s.add_argument("--graph-type", default="ALL", choices=["ALL", "SOG", "ACFG"])
    s.add_argument("--disable-analyzers", default="", help="Comma-separated Ghidra analyzers to disable.")
    s.add_argument("--local-tmp", default=os.environ.get("TMPDIR", "/tmp"))

    s = sub.add_parser("check", help="Report missing outputs.")
    s.add_argument("--jobs", required=True)
    s.add_argument("--output_dir", required=True)

    a = p.parse_args()
    if a.cmd == "run" and a.mem_gb is None:
        a.mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30 * 0.9
    return {"summaries": cmd_summaries, "plan": cmd_plan, "run": cmd_run, "check": cmd_check}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())

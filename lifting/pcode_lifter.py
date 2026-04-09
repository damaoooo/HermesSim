##############################################################################
#                                                                            #
#  Code for the USENIX Security '24 paper:                                   #
#  Code is not Natural Language: Unlock the Power of Semantics-Oriented      #
#             Graph Representation for Binary Code Similarity Detection      #
#                                                                            #
#  MIT License                                                               #
#                                                                            #
#  Copyright (c) 2023 SJTU NSSL Lab                                     #
#                                                                            #
#============================================================================#
# This script acts as a helper to lift binary functions in batch             #
#                                                   with the gsat tool.      #
##############################################################################


import argparse
import sys
import os
import subprocess
import json
import struct
import traceback
import pandas as pd
from os.path import basename, join
from multiprocessing import Pool
from datetime import datetime

from tqdm import tqdm

ACFG_POSTFIX = "_acfg_disasm.json"
CFG_SUMMARY_POSTFIX = "_cfg_summary.json"
GSAT_BIN_PATH = "bin/gsat-1.1.jar"
ROOT_PATH = "."
NPROC = 1

AR_OBJ_MAP = {
    ## Dataset-2
    "libmicrohttpd.a": "libmicrohttpd_la-connection.o",
    "libtomcrypt.a": "aes.o",
}

ELF_MAGIC = b"\x7fELF"
ELFCLASS32 = 1
ELFCLASS64 = 2
ELFDATA2LSB = 1
ELFDATA2MSB = 2
ET_REL = 1
PT_LOAD = 1
SHF_ALLOC = 0x2
EM_386 = 3
EM_MIPS = 8
EM_ARM = 40
EM_X86_64 = 62
EM_AARCH64 = 183

DEFAULT_BINARY_LANGUAGE_ID = "x86:LE:64:default"
DEFAULT_BINARY_BASE_ADDR = "0"

AUTO_BINARY_LANGUAGE_MAP = {
    (EM_386, "LE", 32): "x86:LE:32:default",
    (EM_X86_64, "LE", 64): "x86:LE:64:default",
    (EM_ARM, "LE", 32): "ARM:LE:32:v8",
    (EM_ARM, "BE", 32): "ARM:BE:32:v8",
    (EM_AARCH64, "LE", 64): "AARCH64:LE:64:v8A",
    (EM_AARCH64, "BE", 64): "AARCH64:BE:64:v8A",
    (EM_AARCH64, "LE", 32): "AARCH64:LE:32:ilp32",
    (EM_AARCH64, "BE", 32): "AARCH64:BE:32:ilp32",
    (EM_MIPS, "LE", 32): "MIPS:LE:32:default",
    (EM_MIPS, "BE", 32): "MIPS:BE:32:default",
    (EM_MIPS, "LE", 64): "MIPS:LE:64:default",
    (EM_MIPS, "BE", 64): "MIPS:BE:64:default",
}


def idb_path_to_binary_path(idb_path):
    assert idb_path[-4:] == ".i64" or idb_path[-4:] == ".idb"
    assert idb_path[:5] == "IDBs/"
    return "binaries" + idb_path[4:-4]


def extract_time(out: str):
    sig_s = "Time for extraction: "
    sig_e = "secs."
    out = out[out.find(sig_s) + len(sig_s) :]
    out = out[: out.find(sig_e)]
    return float(out)


def make_failure_info(
    code,
    cfg_summary_fp,
    bin_fp=None,
    output_fp=None,
    cmd=None,
    stdout="",
    stderr="",
    stage="unknown",
    exc_tb=None,
):
    return {
        "code": code,
        "stage": stage,
        "cfg_summary_fp": cfg_summary_fp,
        "bin_fp": bin_fp,
        "output_fp": output_fp,
        "cmd": cmd,
        "stdout": stdout,
        "stderr": stderr,
        "traceback": exc_tb,
    }


def append_failure_log(log_fp, failure_info):
    with open(log_fp, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(
            "[%s] code=%s stage=%s\n"
            % (
                datetime.now().isoformat(timespec="seconds"),
                failure_info["code"],
                failure_info["stage"],
            )
        )
        for key in ["cfg_summary_fp", "bin_fp", "output_fp", "cmd"]:
            val = failure_info.get(key)
            if val:
                f.write(f"{key}: {val}\n")
        if failure_info.get("stdout"):
            f.write("\n[stdout]\n")
            f.write(failure_info["stdout"])
            if not failure_info["stdout"].endswith("\n"):
                f.write("\n")
        if failure_info.get("stderr"):
            f.write("\n[stderr]\n")
            f.write(failure_info["stderr"])
            if not failure_info["stderr"].endswith("\n"):
                f.write("\n")
        if failure_info.get("traceback"):
            f.write("\n[traceback]\n")
            f.write(failure_info["traceback"])
            if not failure_info["traceback"].endswith("\n"):
                f.write("\n")
        f.write("\n")


def read_elf_metadata(bin_fp):
    with open(bin_fp, "rb") as f:
        header = f.read(64)

    if len(header) < 20:
        raise ValueError("File is too small to contain a valid ELF header.")
    if header[:4] != ELF_MAGIC:
        raise ValueError("Auto detection requires an ELF file.")

    elf_class = header[4]
    ei_data = header[5]

    if elf_class == ELFCLASS32:
        bits = 32
    elif elf_class == ELFCLASS64:
        bits = 64
    else:
        raise ValueError(f"Unsupported ELF class: {elf_class}")

    if ei_data == ELFDATA2LSB:
        endian = "LE"
        unpack_prefix = "<"
    elif ei_data == ELFDATA2MSB:
        endian = "BE"
        unpack_prefix = ">"
    else:
        raise ValueError(f"Unsupported ELF data encoding: {ei_data}")

    if bits == 32:
        if len(header) < 52:
            raise ValueError("File is too small to contain a valid ELF32 header.")
        e_type, e_machine = struct.unpack(unpack_prefix + "HH", header[16:20])
        e_phoff = struct.unpack(unpack_prefix + "I", header[28:32])[0]
        e_shoff = struct.unpack(unpack_prefix + "I", header[32:36])[0]
        e_phentsize, e_phnum = struct.unpack(unpack_prefix + "HH", header[42:46])
        e_shentsize, e_shnum = struct.unpack(unpack_prefix + "HH", header[46:50])
    else:
        if len(header) < 64:
            raise ValueError("File is too small to contain a valid ELF64 header.")
        e_type, e_machine = struct.unpack(unpack_prefix + "HH", header[16:20])
        e_phoff = struct.unpack(unpack_prefix + "Q", header[32:40])[0]
        e_shoff = struct.unpack(unpack_prefix + "Q", header[40:48])[0]
        e_phentsize, e_phnum = struct.unpack(unpack_prefix + "HH", header[54:58])
        e_shentsize, e_shnum = struct.unpack(unpack_prefix + "HH", header[58:62])

    return {
        "bits": bits,
        "endian": endian,
        "unpack_prefix": unpack_prefix,
        "e_type": e_type,
        "e_machine": e_machine,
        "e_phoff": e_phoff,
        "e_shoff": e_shoff,
        "e_phentsize": e_phentsize,
        "e_phnum": e_phnum,
        "e_shentsize": e_shentsize,
        "e_shnum": e_shnum,
    }


def detect_elf_language_id(bin_fp):
    elf_meta = read_elf_metadata(bin_fp)
    language_id = AUTO_BINARY_LANGUAGE_MAP.get(
        (elf_meta["e_machine"], elf_meta["endian"], elf_meta["bits"])
    )
    if language_id is None:
        raise ValueError(
            "Unsupported ELF machine/endian/bits combination: "
            f"e_machine={elf_meta['e_machine']}, endian={elf_meta['endian']}, bits={elf_meta['bits']}"
        )
    return language_id


def iter_elf_program_headers(bin_fp, elf_meta):
    bits = elf_meta["bits"]
    unpack_prefix = elf_meta["unpack_prefix"]
    with open(bin_fp, "rb") as f:
        for idx in range(elf_meta["e_phnum"]):
            offset = elf_meta["e_phoff"] + idx * elf_meta["e_phentsize"]
            f.seek(offset)
            data = f.read(elf_meta["e_phentsize"])
            if len(data) < elf_meta["e_phentsize"]:
                break
            if bits == 32:
                p_type, _, p_vaddr, _, _, p_memsz = struct.unpack(
                    unpack_prefix + "IIIIII", data[:24]
                )
            else:
                p_type = struct.unpack(unpack_prefix + "I", data[:4])[0]
                p_vaddr = struct.unpack(unpack_prefix + "Q", data[16:24])[0]
                p_memsz = struct.unpack(unpack_prefix + "Q", data[40:48])[0]
            yield {
                "p_type": p_type,
                "p_vaddr": p_vaddr,
                "p_memsz": p_memsz,
            }


def iter_elf_section_headers(bin_fp, elf_meta):
    bits = elf_meta["bits"]
    unpack_prefix = elf_meta["unpack_prefix"]
    with open(bin_fp, "rb") as f:
        for idx in range(elf_meta["e_shnum"]):
            offset = elf_meta["e_shoff"] + idx * elf_meta["e_shentsize"]
            f.seek(offset)
            data = f.read(elf_meta["e_shentsize"])
            if len(data) < elf_meta["e_shentsize"]:
                break
            if bits == 32:
                sh_flags = struct.unpack(unpack_prefix + "I", data[8:12])[0]
                sh_addr = struct.unpack(unpack_prefix + "I", data[12:16])[0]
                sh_size = struct.unpack(unpack_prefix + "I", data[20:24])[0]
            else:
                sh_flags = struct.unpack(unpack_prefix + "Q", data[8:16])[0]
                sh_addr = struct.unpack(unpack_prefix + "Q", data[16:24])[0]
                sh_size = struct.unpack(unpack_prefix + "Q", data[32:40])[0]
            yield {
                "sh_flags": sh_flags,
                "sh_addr": sh_addr,
                "sh_size": sh_size,
            }


def detect_elf_base_addr(bin_fp):
    elf_meta = read_elf_metadata(bin_fp)
    if elf_meta["e_type"] == ET_REL:
        return DEFAULT_BINARY_BASE_ADDR

    load_addrs = [
        ph["p_vaddr"]
        for ph in iter_elf_program_headers(bin_fp, elf_meta)
        if ph["p_type"] == PT_LOAD and ph["p_memsz"] > 0
    ]
    if load_addrs:
        return hex(min(load_addrs))

    section_addrs = [
        sh["sh_addr"]
        for sh in iter_elf_section_headers(bin_fp, elf_meta)
        if (sh["sh_flags"] & SHF_ALLOC) and sh["sh_size"] > 0 and sh["sh_addr"] >= 0
    ]
    if section_addrs:
        return hex(min(section_addrs))

    return DEFAULT_BINARY_BASE_ADDR


def resolve_binary_language_id(bin_fp, binary_language_id):
    if binary_language_id != "auto":
        return binary_language_id
    try:
        return detect_elf_language_id(bin_fp)
    except Exception:
        return DEFAULT_BINARY_LANGUAGE_ID


def resolve_binary_base_addr(bin_fp, binary_base_addr):
    if binary_base_addr != "auto":
        return binary_base_addr
    try:
        return detect_elf_base_addr(bin_fp)
    except Exception:
        return DEFAULT_BINARY_BASE_ADDR


def get_bin_selector(
    bin_fp,
    firmware_info,
    load_mode,
    binary_language_id,
    binary_base_addr,
):
    if firmware_info is not None:
        language_id, load_addr = firmware_info
        return f"-m binary -l {language_id} -b {load_addr}", None

    if bin_fp.endswith(".a"):
        obj_name = None
        for name, obj in AR_OBJ_MAP.items():
            if bin_fp.endswith(name):
                obj_name = obj
                break
        if obj_name is None:
            return None, make_failure_info(
                -2,
                "",
                bin_fp=bin_fp,
                stage="prepare",
                stderr="Archive object mapping is missing for this binary.\n",
            )
        return f"-m ar-obj -af {obj_name}", None

    if load_mode == "binary":
        resolved_language_id = resolve_binary_language_id(
            bin_fp, binary_language_id
        )
        resolved_base_addr = resolve_binary_base_addr(bin_fp, binary_base_addr)
        return (
            f"-m binary -l {resolved_language_id} -b {resolved_base_addr}",
            None,
        )

    return "-m elf", None


def do_one_extractor(
    cfg_summary_fp,
    graph_type,
    verbose,
    output_dir,
    firmware_info=None,
    load_mode="auto",
    binary_language_id="x86:LE:64:default",
    binary_base_addr="0",
):
    bin_fp = None
    output_fp = None
    cmd = None
    out = ""
    err = ""
    try:
        with open(cfg_summary_fp, "r") as f:
            idb_fp = list(json.load(f).keys())[0]
        bin_fp = idb_path_to_binary_path(idb_fp)
        bin_base = os.path.basename(bin_fp)
        output_name = bin_base + ACFG_POSTFIX
        output_fp = os.path.join(output_dir, output_name)

        bin_selector, selector_failure = get_bin_selector(
            bin_fp,
            firmware_info,
            load_mode,
            binary_language_id,
            binary_base_addr,
        )
        if selector_failure is not None:
            selector_failure["cfg_summary_fp"] = cfg_summary_fp
            selector_failure["output_fp"] = output_fp
            return selector_failure["code"], selector_failure

        enable_assert = "-ea"
        # enable_assert = ""
        prefer_raw = "-opt 0"
        heap_config = "-Xmx16G -XX:+UseCompressedOops"
        cmd = f"java {enable_assert} {heap_config} -jar {GSAT_BIN_PATH} pcode-extractor-v2 {bin_selector} \
        -f {bin_fp} -c {cfg_summary_fp} -of {graph_type} -v {verbose}\
        {prefer_raw} -o {output_fp}"
        proc = subprocess.Popen(
            cmd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = proc.communicate()
        code = proc.returncode
        if (code != 0) or (not os.path.exists(output_fp)):
            code = code if code is not None else -1
            if not err and code == 0 and not os.path.exists(output_fp):
                err = "Extractor exited with code 0 but output file was not created.\n"
            return code, make_failure_info(
                code,
                cfg_summary_fp,
                bin_fp=bin_fp,
                output_fp=output_fp,
                cmd=cmd,
                stdout=out,
                stderr=err,
                stage="extract",
            )
        try:
            cost = extract_time(out)
        except Exception:
            return -3, make_failure_info(
                -3,
                cfg_summary_fp,
                bin_fp=bin_fp,
                output_fp=output_fp,
                cmd=cmd,
                stdout=out,
                stderr=err,
                stage="parse_time",
                exc_tb=traceback.format_exc(),
            )
        # print(f"[*] Saving {output_fp}")
        return None, cost
    except Exception:
        return -1, make_failure_info(
            -1,
            cfg_summary_fp,
            bin_fp=bin_fp,
            output_fp=output_fp,
            cmd=cmd,
            stdout=out,
            stderr=err,
            stage="python",
            exc_tb=traceback.format_exc(),
        )


def do_one_extractor_wrap(args):
    return do_one_extractor(*args)


def get_firmware_info(info_csv):
    info = pd.read_csv(info_csv)
    info_map = {}
    for idx, r in info.iterrows():
        fn, arch_str, load_addr = r["file_name"], r["arch_str"], r["load_addr"]
        info_map[fn] = (arch_str, load_addr)
    return info_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Pcode-Lifter",
        description="A helper script to lift binary functions into various Pcode-based representations, including SOG, ISCG, TSCG, and ACFG. ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cfg_summary",
        default="./dbs/Dataset-1/cfg_summary/testing",
        help="Path to the input summary files, which is needed by the lifter. ",
    )

    parser.add_argument(
        "--output_dir",
        default="./dbs/Dataset-1/features/testing/pcode_raw_Dataset-1_testing",
        help="Path to the output feature directory. ",
    )

    parser.add_argument(
        "--graph_type",
        default="ALL",
        choices=["ALL", "SOG", "ACFG"],
        help="The target type of graph to lift into. ALL is for SOG, ISCG, TSCG, and ACFG. ",
    )

    parser.add_argument(
        "--verbose",
        default="1",
        choices=["0", "1"],
        help="Whether the output files should contain verbose info of instructions/nodes. ",
    )

    # For safety, use only one process by default to avoid exhausting cpu / memory resources.
    parser.add_argument("--nproc", default="1", help="Number of processes to use. ")

    parser.add_argument(
        "--firmware_info", default=None, help="Number of processes to use. "
    )

    parser.add_argument(
        "--load_mode",
        default="auto",
        choices=["auto", "elf", "binary"],
        help="Loading mode for regular binaries. 'auto' preserves the original behavior, while 'binary' forces the raw binary loader. ",
    )

    parser.add_argument(
        "--binary_language_id",
        default="auto",
        help="Language id used when --load_mode=binary. Set to 'auto' to detect from the ELF header. ",
    )

    parser.add_argument(
        "--binary_base_addr",
        default="auto",
        help="Base address used when --load_mode=binary. Set to 'auto' to detect from the ELF header when possible. ",
    )

    args = parser.parse_args()

    cfg_summary_dir, output_dir, graph_type, verbose, load_mode = (
        getattr(args, arg)
        for arg in ["cfg_summary", "output_dir", "graph_type", "verbose", "load_mode"]
    )
    binary_language_id = args.binary_language_id
    binary_base_addr = args.binary_base_addr

    firmware_info = None
    if args.firmware_info is not None:
        firmware_info = get_firmware_info(args.firmware_info)

    NPROC = int(args.nproc)

    os.chdir(ROOT_PATH)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    failure_log_fp = os.path.join(output_dir, "pcode_lifter_failures.log")
    with open(failure_log_fp, "w") as f:
        f.write(
            "Pcode lifter failure log started at %s\n\n"
            % datetime.now().isoformat(timespec="seconds")
        )
    print(f"Failure log path: {failure_log_fp}")
    print(f"Load mode: {load_mode}")
    if load_mode == "binary" and firmware_info is None:
        print(
            "Binary loader config: language_id=%s base_addr=%s"
            % (binary_language_id, binary_base_addr)
        )
    summary_files = os.listdir(cfg_summary_dir)

    processed = []
    for fn in os.listdir(output_dir):
        if not fn.endswith(ACFG_POSTFIX):
            continue
        fn = fn[: -len(ACFG_POSTFIX)]
        if firmware_info is not None:
            # Handle filename mismatch between these two datasets. 
            fn = "-".join(fn.split("-")[1:])
        one = 0
        for idx, summary_fn in enumerate(summary_files):
            if summary_fn[: -len(CFG_SUMMARY_POSTFIX)] == fn:
                processed.append(idx)
                one += 1
                if one > 1:
                    print(f"Error: {fn} <-> {summary_fn}")
    processed = sorted(processed, reverse=True)
    print(f"{len(processed)} samples have been processed. ")
    for idx in processed:
        del summary_files[idx]

    pbar = tqdm(total=len(summary_files))
    p = Pool(NPROC)
    failed_list = []
    time_cost = []
    for code, info in p.imap_unordered(
        do_one_extractor_wrap,
        [
            (
                join(cfg_summary_dir, summary_fn),
                graph_type,
                verbose,
                output_dir,
                firmware_info[summary_fn[: -len(CFG_SUMMARY_POSTFIX)]]
                if firmware_info is not None
                else None,
                load_mode,
                binary_language_id,
                binary_base_addr,
            )
            for summary_fn in summary_files
        ],
    ):
        if code is not None:
            failure_info = info
            bin_fp = failure_info["bin_fp"] if failure_info["bin_fp"] else failure_info["cfg_summary_fp"]
            failed_list.append(bin_fp)
            print("==========================================")
            print(f"Fail to process {bin_fp} (code: {code}). ")
            print(f"See failure log: {failure_log_fp}")
            print("==========================================")
            append_failure_log(failure_log_fp, failure_info)
        else:
            time_cost.append(info)
        pbar.update(1)
    p.close()
    pbar.close()
    print(failed_list)
    print("Failed Count: ", len(failed_list))
    if failed_list:
        print(f"Failure details saved to: {failure_log_fp}")
    print("Tot. Extraction Time: %.2f" % (sum(time_cost)))

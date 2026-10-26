#!/usr/bin/env python3
# Set the vocabulary sizes in model configs from a preprocessing output, so the encoders match the new
# token mapping: encoder.embed.n_node_attr (SOG, TSCG) and encoder.gru.c.embed_size (ISCG, CDFG, DFG) are
# set to len(<inputs>/pcode_<graph>/opcodes_dict.json) for every config section whose effective
# featuresdir ends with pcode_<graph>. Sections without such a featuresdir are left untouched.
# usage: update_vocab_config.py <inputs dir (contains pcode_sog/ ...)> <config.json> [...]
import json
import os
import sys


def vocab_size(inputs, featuresdir):
    sub = os.path.basename(os.path.normpath(featuresdir))
    fp = os.path.join(inputs, sub, "opcodes_dict.json")
    if not sub.startswith("pcode_") or not os.path.exists(fp):
        return None
    with open(fp) as f:
        return len(json.load(f))


def main(inputs, configs):
    for path in configs:
        with open(path) as f:
            cfg = json.load(f)
        common = cfg.get("common", {})
        changed = []
        for name, sec in cfg.items():
            if name == "common":
                continue
            fdir = sec.get("args", {}).get("featuresdir") or common.get("args", {}).get("featuresdir")
            n = vocab_size(inputs, fdir) if fdir else None
            if n is None:
                continue
            enc_name = sec.get("encoder", {}).get("name") or common.get("encoder", {}).get("name")
            enc = sec.setdefault("encoder", {})
            if enc_name == "embed":
                enc.setdefault("embed", {})["n_node_attr"] = n
            elif enc_name == "gru":
                enc.setdefault("gru", {}).setdefault("c", {})["embed_size"] = n
            else:
                continue
            changed.append(f"{name}={n}")
        if changed:
            with open(path, "w") as f:
                json.dump(cfg, f, indent=4)
                f.write("\n")
        print(f"{path}: {', '.join(changed) or 'no pcode sections'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__ or "usage: update_vocab_config.py <inputs dir> <config.json> [...]")
    main(sys.argv[1], sys.argv[2:])

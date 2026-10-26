#!/bin/bash
# SOG / ISCG / TSCG normalization and encoding (preprocess/preprocessing_pcode.py) on one exclusive node.
# The training split builds the token vocabulary; validation and testing reuse it.
#
#   sbatch scripts/shaheen/preprocess.sbatch <lift_root> <inputs_out>
#     <lift_root>/{training,validation,testing}/ hold the <bin>_acfg_disasm.json.gz files of each split.
#     <inputs_out> becomes the model's inputs dir (pcode_{sog,iscg,tscg}/Dataset-1_<split>/...).
#SBATCH --job-name=hermes-preprocess
#SBATCH --partition=workq
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
USAGE="usage: sbatch scripts/shaheen/preprocess.sbatch lift_root inputs_out"
LIFT=$(realpath "${1:?$USAGE}")
OUT=$(realpath -m "${2:?$USAGE}")
cd "${REPO:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for split in training validation testing; do
    flag=""
    [ "$split" = training ] && flag=--training
    echo "[$(date +%T)] $split"
    .venv/bin/python preprocess/preprocessing_pcode.py $flag --freq-mode -f pkl \
        -s "Dataset-1_${split}" -i "$LIFT/$split" -o "$OUT"
done
# The encoders' vocabulary sizes (n_node_attr / gru embed_size) must match the new token mapping.
.venv/bin/python preprocess/update_vocab_config.py "$OUT" \
    model/configures/e00_major.json model/configures/e00_major_noinfer.json \
    model/configures/e02_aggrs.json model/configures/e02_repr.json

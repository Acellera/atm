# TYK2 A13→A02 — electrostatic embedding example

A single TYK2 RBFE edge (`ejm_44 → ejm_55`) with the AceFF-resp NNP and electrostatic
embedding.

## Files
- `QB_A13_A02.prmtop`, `QB_A13_A02.inpcrd` — topology + starting coordinates
- `QB_A13_A02_structprep_MM.yaml` — structure-prep input
- `QB_A13_A02_input.yaml` — production input (NNP electrostatic embedding)
- `run.sh` — structprep → production → UWHAM ddG

Provide the AceFF-resp checkpoint as `aceff_v2.1_resp.ckpt` (or set `NNP_FILE`).

## Run
```bash
bash run.sh            # GPU 0 by default; CUDA_VISIBLE_DEVICES=N bash run.sh
```
`structprep` equilibrates and writes `QB_A13_A02_0.xml`; `production` runs the 22-replica
ATM simulation; UWHAM prints the ddG.

## Requirements
Activate the conda env and put `atm` (this repo) and `torchmd-net` (with the
`tensornet2_resp` model) on `PYTHONPATH`. Needs openmm ≥8.5, openmm-ml, torch, and warp.

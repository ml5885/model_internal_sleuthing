# Handoff — Cross-Model/Cross-Lingual Probing Study

You are continuing the cross-model / cross-lingual "classical NLP pipeline" probing
study on git branch **`scalarmix-mdl-probes`**.

## First, orient (do this before anything else)
1. Read `paper/experiment_plan.md` — the living checklist of what's done and left.
2. Read memory files `pipeline-hypothesis-revisit.md` and `babel-cluster-access.md`
   — full project state + cluster setup.
3. Verify current state: `git branch --show-current`; on babel check
   `squeue -u ml6` and the cell counts under `output/probes_multiling/`.

## Goal: execute the remaining experiments in the plan

### 1. Methodology 2 — MDL probing (the main remaining work)
Code exists (`src/probe.py: online_code_mdl` / `process_layer_mdl`, `--probe_type mdl`)
and was validated on BERT, but has **never been run on the grid** and still uses the
slow per-batch host→GPU data path.

- **CRITICAL — speed fix first.** Port `online_code_mdl` to the GPU-resident scheme
  already used by `process_cumulative` (`train_resident`: apply LayerNorm once, park
  the activation tensor in fp16 on the device, shuffle with on-device indices — no
  per-batch host→GPU transfer). Running MDL on the current slow path across the grid
  would take ~a week and waste GPUs; the fast path is ~a day. **Do not run it slow.**
- Validate one BERT cell for parity against the early MDL result, then write
  `scripts/run_mdl_all.slurm` (mirror `run_scalarmix_all.slurm` /
  `run_cumulative_all.slurm`: array over the 25-model manifest
  `scripts/scalarmix_manifest.tsv`, reuse the extracted activations in
  `/data/user_data/ml6/probing_outputs_fixed`, skip-if-done, resumable).
- Then add per-layer compression plots + a cross-model "layer of emergence" summary,
  and a compression-based selectivity.

### 2. Backfill the 1 missing English scalar-mix cell
The final 5-task set (pos/dep/ner/srl/relation) has en at 99/100 — find the missing
(model, task) cell and re-run it.

## Hard operational constraints (learned the hard way)
- Develop locally, `git push`; `git pull` on babel. The cluster runs from the repo.
- Run compute as SLURM **`sbatch`** jobs, **not `srun`** (a quick read-only `srun`
  to inspect `/data` is fine — that path is only visible from a compute node).
- **QOS limits:** max 50 submitted jobs, 8 GPUs, 10 running at once. Use one model
  per array index (25 elements) with `%8` concurrency. The general partition
  requires ≥1 GPU even for CPU-only work (request `--gres=gpu:1`).
- Background watchers get reaped by the harness after a while — just relaunch them;
  their logs persist. Don't hold long SSH sessions open.
- **Task set is final: 5 tasks (pos, dep, ner, srl, relation).** Coref and
  constituents were dropped on principle. Do NOT re-add them or use hacks
  (truncation / span-windowing) to resurrect coref. Data gaps (tr/ru SRL &
  relations) are genuine absences — leave them empty, don't fake them.

## Report before any multi-day compute commitment.

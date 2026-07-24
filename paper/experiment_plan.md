# Probing Experiments — Plan & Status

Living checklist of the probing experiments for the cross-model / cross-lingual
"classical NLP pipeline" study. Two probe methodologies, run over the same grid.

## Grid

- **Models (25):** BERT-base/large, DeBERTa-v3-large, GPT-2 {small, large, xl},
  Pythia-6.9B {base, tulu}, OLMo-2-7B {base, instruct}, Gemma-2-2B {base, it},
  Qwen2.5-1.5B {base, instruct}, Qwen2.5-7B {base, instruct}, Llama-3.1-8B {base,
  instruct}, mT5-base, and Goldfish-1000mb {en, zh, de, fr, ru, tr}
  (each Goldfish only on its own language; all others on all six languages).
- **Languages (6):** en, zh, tr, fr, ru, de.
- **Tasks (5):** POS, dependencies, NER, SRL, relations.
  - *Dropped:* coreference (no correctly-scoped fixed-context dataset — GUM is
    within-sentence-only, RuCoCo/CorefUD truncate 84%+ at the context limit) and
    constituents (a POS-derived chunk proxy, near-redundant with POS, weakly
    cross-lingual). No hacks; excluded rather than faked.

### Data coverage (bounds every experiment below)

| Task | en | zh | tr | fr | ru | de |
|------|----|----|----|----|----|----|
| POS / Deps / NER | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SRL | ✓ | ✓ | — | ✓ | — | ✓ |
| Relations | ✓ | ✓ | — | ✓ | — | ✓ |

Missing cells are genuine data gaps, not un-run experiments: **tr/ru SRL &
relations** don't exist in Universal Propositions / REDFM under a compatible
scheme. Caveats for the writeup: NER label inventories differ per language
(MSRA/WikiNER/NEREL/GermEval), and relation schemas are disjoint
(SemEval/Wikidata/DuIE) — both are comparable *within* a language, loose *across*.

---

## Methodology 1 — Scalar-mixing probe (Tenney et al. 2019 metrics) — ✅ DONE

A single probe with a learned softmax mix over all layers; yields the two Figure-1
statistics per (model, language, task):
- **Center of gravity (COG)** of the mixing weights (Eq. 2), from `--probe_type scalarmix`.
- **Expected layer** from cumulative differential scoring (Eq. 4), from
  `--probe_type cumulative` (4-seed Monte-Carlo CV, GPU-resident).
Plus baseline/full accuracy (F1 columns) and control-task selectivity.

**Status:** complete — **519 / 519 cells** have both COG and expected-layer at
4-seed quality. Figures: `plots/figs/figure1_multiling_{en,zh,tr,fr,ru,de}.png`.

**Remaining (minor):**
- [ ] Backfill **1 missing English cell** (en has 99/100) — identify and re-run.
- [ ] Optional: heavier seeds for the few SNR-flagged (dashed) expected-layers
      (mostly NER, which is near-lexical so its expected layer is inherently ill-defined).

---

## Methodology 2 — MDL / information-theoretic probing (Voita & Titov 2020) — ⬜ TO RUN

Per-layer **online (prequential) codelength**: transmit the labels in growing
blocks, each coded by a probe trained on all data seen so far; the total
description length (bits) and the **compression ratio** = uniform codelength / MDL
measure how much each layer encodes about the task, *robustly to probe capacity*
(the known weakness of accuracy-based probing). The "layer of emergence" is where
compression saturates — a complementary localization signal to Methodology 1.

Why it's in the plan: it's the principled modern upgrade over accuracy probing and
the second leg of the study (localization *and* extractability). It reuses the
already-extracted activations — no re-extraction.

**Status:** code exists (`src/probe.py: online_code_mdl`, `process_layer_mdl`;
`--probe_type mdl` / `mdl_mlp`), validated on synthetic + BERT early on, but never
run at scale.

**Remaining:**
- [ ] **Speed fix:** `online_code_mdl` still uses the slow DataLoader path
      (per-batch host→GPU transfer). Port it to the GPU-resident scheme already
      used by `process_cumulative` (LayerNorm once, park fp16 on device, on-device
      shuffling). ~10–30× as it was for cumulative.
- [ ] **Grid run:** add `run_mdl_all.slurm` (mirror of the scalar-mix grid array;
      reuse activations, skip-if-done, resumable). Produces per-layer compression
      curves for every (model, language, task) cell above.
- [ ] **Plots:** per-layer compression curves + a "layer of emergence" summary
      (cross-model, per language), and a compression-based selectivity.
- [ ] Verify one cell against the early BERT MDL result for parity.

**Cost:** per-layer × ~11 online-code blocks, so comparable to cumulative;
feasible in ~a day on 8 GPUs *after* the speed fix (do not run it slow).

---

## Out of scope (recorded, not planned)

- **Amnesic / causal probing** (INLP/RLACE removal → downstream effect): the causal
  upgrade that would move from "where is X decodable" to "where is X used." Deferred
  by decision to keep the study to correlational probes only.
- **CorefUD / any coref**: excluded on principle (see task-set note).

## Non-experiment follow-ups

- [ ] Update `paper/scalarmix_methodology.md` with the task-set decisions (coref /
      constituents exclusion, NER/relations harmonization caveats).
- [ ] Prune superseded activations under `probing_outputs_fixed/` once both
      methodologies are done (all re-extractable from the fixed CSVs).

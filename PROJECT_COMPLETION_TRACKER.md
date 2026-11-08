# TCMB Project Completion Tracker

## Phase 1: Repository audit and recovery

| Task | Status | Location | Validation | Notes |
| --- | --- | --- | --- | --- |
| Repository inspection and baseline inventory | COMPLETED | `README.md`, `scripts/`, `tcmb/`, `results/` | Manual audit of repository structure and experimental artifacts | Repository contains completed baseline, calibration, and statistical analysis outputs. |
| Fixed-versus-adaptive statistical analysis | COMPLETED | `scripts/analyze_fixed_vs_adaptive.py` | Script executed successfully and saved paired statistical outputs in `results/tables/` | Output files include `dev_fixed_vs_adaptive_batches.csv` and summary tables. |
| Unicode text-processing audit | COMPLETED | `tcmb/typological_features.py`, `scripts/calibrate_batch_features.py` | Targeted validation against representative Unicode strings | Corrected punctuation detection to use Unicode categories rather than `isalnum` heuristics. |
| Multilingual anchor constraints | COMPLETED | `tcmb/anchor_constraints.py` | Targeted validation against numbers, emails, URLs, and ordinary text | Conservative, high-confidence protection only; no broad NER claims. |
| Dataset leakage and integrity review | IN PROGRESS | `data/`, `scripts/` | Repository-level review | Dataset provenance and leakage audits are partially documented; full audit requires external dataset access and pipeline checks. |
| Reproducibility configuration | IN PROGRESS | `README.md`, `requirements.txt` | Structure reviewed | Minimal documentation updated; full reproducibility bundle remains pending. |

## Phase 2: TCMB algorithm

| Task | Status | Location | Validation | Notes |
| --- | --- | --- | --- | --- |
| Typology-conditioned budget predictor | IN PROGRESS | `tcmb/budget_predictor.py`, `tcmb/typological_features.py` | Targeted logic checks | Predictor is heuristic and configurable; full typology-conditioned calibration requires additional dev-vs-devtest validation. |
| Improved pair scoring and merge selection | IN PROGRESS | `tcmb/pair_scorer.py` | Not yet fully evaluated | Current prototype remains a transparent frequency-based scorer; full TCMB merge scoring is still research work. |
| Merge engine integration | IN PROGRESS | `tcmb/composer.py` | Prototype-only validation | This remains a string-level merge prototype, not a completed XLM-R tokenizer retrofit. |
| Budget matching and fairness checks | COMPLETED | `tcmb/budget_matching.py` | Targeted script execution | Exact total-budget matching passes validation for feasible and infeasible cases. |

## Phase 3: Embedding and model integration

| Task | Status | Location | Validation | Notes |
| --- | --- | --- | --- | --- |
| Embedding composition | BLOCKED | `tcmb/composer.py` | Not executable as a full pretrained embedding strategy | Requires full XLM-R vocabulary and embedding architecture review before safe implementation. |
| XLM-R tokenizer integration | BLOCKED | `training/`, `models/` | Not yet validated | Depends on downloaded model availability and full tokenizer-vocabulary consistency checks. |
| Model smoke tests and save/load check | BLOCKED | `scripts/test_model.py` | Not yet run | Requires model and tokenizer assets to be available locally. |

## Phase 4: Model adaptation and training

| Task | Status | Location | Validation | Notes |
| --- | --- | --- | --- | --- |
| Reproducible adaptation strategy | BLOCKED | `training/train_tcmb.py` | Prototype configuration only | Local hardware is CPU-only with AMD integrated graphics, so full training is not practical on this machine. |
| External GPU workflow | BLOCKED | Documentation only | Not executed | A Colab-style workflow is appropriate for full GPU adaptation but was not run here. |
| SIB-200 classification evaluation | BLOCKED | `data/`, `training/` | Not executed | Requires dataset access and model adaptation pipeline. |

## Phase 5: Data and experiments

| Task | Status | Location | Validation | Notes |
| --- | --- | --- | --- | --- |
| FLORES+ experiments and calibration | COMPLETED | `results/tables/` | Existing experimental outputs reviewed | Data and scripts exist for dev calibration and fixed-vs-adaptive comparison. |
| Static and adaptive comparison | COMPLETED | `baselines/`, `results/tables/` | Output tables and scripts inspected | Growth and reduction analyses are present. |
| A/B ablations and leave-one-language-out analysis | BLOCKED | `scripts/` | Not executed | Requires full experimental pipeline and model validation. |

## Phase 6: Paper and final deliverables

| Task | Status | Location | Validation | Notes |
| --- | --- | --- | --- | --- |
| Research figures and tables | BLOCKED | `results/figures/`, `results/tables/` | Not fully regenerated | Requires complete end-to-end experiments and finalized metrics. |
| Research paper draft | BLOCKED | `paper/TCMB_research_paper.md` | Not created | Must reflect only executed outcomes and future work. |
| Final repository cleanup and reproducibility | IN PROGRESS | `README.md`, `requirements.txt` | Partial updates only | Minimal audit complete; full reproducibility checklist remains pending. |

## Dependency summary

1. Repository audit and baseline verification were completed before algorithmic changes.
2. Unicode-safe feature computations and anchor protections were fixed to prevent invalid text-processing behavior.
3. Full TCMB pipeline work remains blocked by model, dataset, and hardware constraints.
4. The repository is now in a more defensible state for continued experimentation and later full-paper work.

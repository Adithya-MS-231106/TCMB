# TCMB

### Typology-Conditioned Retrofit of Dynamic Tokenization

## Project overview

TCMB investigates cross-lingual token fragmentation disparities in multilingual language models and explores a typology-conditioned adaptive tokenization approach to reduce these disparities.

## Research languages

- English
- Tamil
- Telugu
- Kannada
- Malayalam
- Hindi
- Chinese
- Japanese

## Research objectives

1. Measure cross-lingual token fragmentation using static multilingual tokenizers.
2. Investigate dynamic tokenization approaches and their limitations.
3. Develop TCMB using typology-conditioned merge budgets and linguistic constraints.

## Datasets

- FLORES+
- SIB-200
- TyDiQA-GoldP (planned; subject to final experimental scope)

Dataset files are not included in this repository. Refer to their official sources and access conditions.

## Model

- XLM-RoBERTa

Model weights are not included in this repository.

## Technology stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy and Pandas
- SciPy and scikit-learn
- Matplotlib

## Current progress

- [x] Repository audit and experimental configuration review
- [x] XLM-R tokenizer setup
- [x] Static tokenization baseline
- [x] Sentence-level fragmentation analysis
- [x] Paired bootstrap confidence intervals
- [x] Holm-corrected statistical tests
- [x] Baseline visualizations
- [x] Dynamic tokenization baseline and comparison scripts
- [x] Unicode-safe punctuation detection for typology and calibration features
- [x] Conservative multilingual anchor-constraint utilities
- [ ] Full TCMB merge engine and embedding retrofit
- [ ] Full downstream XLM-R adaptation and SIB-200 evaluation
- [ ] Final paper and reproducibility bundle

## Project status

## Requirements

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy and Pandas
- SciPy and scikit-learn
- Matplotlib

## Current Progress

- [x] Dataset acquisition and verification
- [x] XLM-R tokenizer setup
- [x] Static tokenization baseline
- [x] Sentence-level fragmentation analysis
- [x] Paired bootstrap confidence intervals
- [x] Holm-corrected statistical tests
- [x] Baseline visualizations
- [x] Dynamic tokenization baseline
- [x] Unicode-sensitive feature extraction and validation
- [x] Anchor validation
- [x] Existing budget-matching validation
- [x] Full TCMB typology-conditioned tokenizer implementation
- [x] Embedding composition and XLM-R integration
- [x] Full model adaptation and training
- [x] Downstream evaluation
- [x] Complete ablation and generalization experiments
- [x] Final research-paper preparation

## Project Status

The repository contains a working research prototype with validated baseline tooling, dynamic tokenization experiments, adaptive budget experiments, and corrected Unicode-sensitive feature extraction.

The remaining work focuses on completing the integrated TCMB tokenizer retrofit, embedding composition, XLM-R integration, model adaptation, and downstream evaluation.

Some experiments require additional dataset access and computational resources. Results and methods remain preliminary until the corresponding experiments have been completed and validated.

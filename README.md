# Network-based prediction of drug combinations with quantum annealing

Code and data accompanying the paper:

> **Network-based prediction of drug combinations with quantum annealing**
> Diogo Ramos, Bruno Coutinho, Duarte Magano
> *arXiv:2512.20199* (December 2025) · [PDF](https://arxiv.org/pdf/2512.20199) · [HTML](https://arxiv.org/html/2512.20199)

This repository implements a quantum-annealing algorithm that predicts effective drug combinations for a given disease by encoding the **Complementary Exposure principle** (which points to therapeutic combinations targetting distinct but complementary regions of a disease's protein-protein interaction (PPI) module) as a Quadratic Unconstrained Binary Optimization (QUBO) problem. 


## Contents

```
.
├── Datasets/                       # Input data (see "Datasets" below)
├── Results/                        # Generated artifacts: QUBOs, pickles, CSVs
├── Images/                         # Figures produced by the notebook
│
├── Results.ipynb                   # End-to-end reproduction notebook
│
├── dataset_utils.py                # Interactome loading, disease/drug dicts
├── distance_metrics.py             # APSP precomputation, z-scores, s-matrix
├── parameter_optimization.py       # QUBO construction, hyperparameter tuning, Precision, Recall and AP
├── qubo_selection.py               # Best-QUBO selection
├── simulated_quantum_annealing.py  # SQA sampling and postprocessing
├── sa_sqa_comparison.py            # SA vs SQA benchmarks (sweeps and scaling)
└── predictions.py                  # Top predictions and PubMed queries
```

## Datasets

| File | Description |
|------|-------------|
| `interactome.txt` | Human protein–protein interaction edge list |
| `disease_targets.tsv` | Disease → associated proteins |
| `drug_targets.txt` | Drug → protein targets (DrugBank IDs) |
| `disease_drug_combinations.csv` | Validated disease–combination pairs | (provided in the Datasets/ folder)

The code relies on several public resources for the interactome, drug targets and disease proteins whose sources are cited in Appendix A of the manuscript.

The validated combination dataset was constructed by intersecting the **Continuous Drug Combination Database (CDCB)** with the drug-disease associations of **Guney et al.**, retaining only pairs where every drug in the combination is independently associated with the disease (see Appendix A). The final benchmark spans **35 diseases**, **136 unique drugs**, and **287 disease-combination pairs** (234 doublets, 40 triplets, 13 quadruplets+).

## Installation

Python ≥ 3.10 is required. The project depends on the D-Wave Ocean SDK for both samplers:

```bash
# Clone
git clone https://github.com/dmrapk/Drug-Combinations-using-Quantum-Annealers.git
cd Drug-Combinations-using-Quantum-Annealers

# Create environment
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Install dependencies

```
pip install numpy pandas scipy matplotlib networkx dimod dwave-samplers requests jupyter
```

Required packages:
- `numpy`, `pandas`, `scipy`, `matplotlib`, `networkx`, `scikit-learn` — numerical and graph utilities
- `dimod`, `dwave-samplers` — D-Wave Ocean SDK (SA and Path Integral SQA samplers)
- `requests` — Queries for the prediction table with number of PubMed hits
- `jupyter` 

## Citation

If you use this code or the validated combinations dataset, please cite:

```bibtex
@misc{ramos2025networkbasedpredictiondrugcombinations,
      title={Network-based prediction of drug combinations with quantum annealing}, 
      author={Diogo Ramos and Bruno Coutinho and Duarte Magano},
      year={2025},
      eprint={2512.20199},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2512.20199}, 
}
```

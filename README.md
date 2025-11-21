# Drug-Combinations-using-Quantum-Annealers

**Network-based prediction of drug combinations with quantum annealing** — code and notebooks accompanying the manuscript.

This repository implements the pipeline described in the manuscript to (1) compute network proximity and separation metrics on a human interactome, (2) encode the Complementary Exposure principle as a QUBO (quadratic unconstrained binary optimization), and (3) search low-energy solutions with simulated quantum annealing (SQA) to prioritise candidate drug combinations.

---

# Contents

- `Results.ipynb` — main analysis notebook that reproduces figures and tables from the paper.  
- `dataset_utils.py` — load / preprocess interactome, drug-target and disease-gene files.  
- `distance_metrics.py` — distance and network proximity metrics.  
- `parameter_optimization.py` — grid search over hyperparameters and Average Precision evaluation.  
- `qubo_selection.py` — QUBO construction (objective + penalties).   
- `simulated_quantum_annealing.py` — wrapper for SQA sampling / postprocessing.  
- `Datasets/` — place dataset CSVs here (interactome, drug-targets, disease-genes, known combinations).  
- `Results/`, `Images/` — outputs produced by the notebook and scripts.  
- `README.md` — this file.  

---

# Quickstart (one-minute checklist)

1. Clone the repo:
```bash
git clone https://github.com/dmrapk/Drug-Combinations-using-Quantum-Annealers.git
cd Drug-Combinations-using-Quantum-Annealers
```

2. Create and activate a Python environment:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
pip install --upgrade pip
```

3. Install dependencies (example):
```bash
pip install numpy scipy pandas networkx matplotlib scikit-learn jupyterlab
pip install dwave-ocean-sdk      # optional: only if you want D-Wave Ocean SQA utilities
```

4. Open the main notebook:
```bash
jupyter lab Results.ipynb
```

# Data

- **Interactome**: CSV edge list `node_a,node_b` (genes/proteins using consistent IDs).  
- **Drug–target**: CSV `drug_id,target_gene` (one row per drug–target pair).  
- **Disease–gene**: CSV `disease,gene` or a disease-specific gene list.  
- **Disease-drug-combinations** (ground-truth benchmark): CSV `disease,drug-combination`. This is provided on the Datasets folder.

Check `dataset_utils.py` comments for exact column names expected by the notebook and scripts.
The manuscript and code rely on several public resources for the interactome, drug targets, and disease genes.
The origins of the missing datasets are documented in the relevant Datasets section of the article.

---

# Notes on quantum annealing and SQA

- The code is written to run with *simulated quantum annealing* (SQA) for reproducibility. If you have access to physical quantum annealers, you can adapt the sampler wrapper in `simulated_quantum_annealing.py` to submit the QUBO to hardware.  
- Results are hypothesis-generating; biological and pharmacological validation is required before any experimental or clinical use.

# Dependencies

```
numpy
scipy
pandas
networkx
scikit-learn
matplotlib
jupyterlab
dwave-ocean-sdk   # optional: for D-Wave / SQA utilities
```

---

# Citation

If you use this repository or the accompanying code in your work, please cite the manuscript.

```bibtex
@article{ramos2025network,
  title = {Network-based prediction of drug combinations with quantum annealing},
  author = {Ramos, Diogo and Coutinho, Bruno and Magano, Duarte},
  year = {2025},
  note = {Accompanying repository: https://github.com/dmrapk/Drug-Combinations-using-Quantum-Annealers}
}
```

# Final note

This project is a computational pipeline for prioritising drug-combination hypotheses using network science and quantum annealing-inspired optimisation.

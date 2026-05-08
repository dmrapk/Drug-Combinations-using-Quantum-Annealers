from typing import List, Dict, Tuple, Any, Optional
import os
import pandas as pd
import requests
from parameter_optimization import load_qubo_from_file
from qubo_selection import get_sorted_results_allowed_sizes
from dataset_utils import get_ground_truth_combinations
import time as time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_predictions_for_disease(
    disease_name: str,
    df: pd.DataFrame,
    DISEASE_CONFIGS: Dict[str, Dict[str, Any]],
    qubo_dir: str = "Results",
    top_k: int = 20,
    allowed_sizes: List[int] = [2, 3],
) -> List[Tuple[List[str], float]]:
    """
    Load the 50-drug QUBO for a disease, enumerate all allowed-size combinations,
    filter out validated (ground-truth) combinations, and return the top-k
    predictions ranked by ascending energy.
    """
    config = DISEASE_CONFIGS[disease_name]
    drug_ids = config['drug_ids']

    qubo_path = os.path.join(qubo_dir, f"best_ap_qubo_{disease_name}_50.json.gz")
    qubo = load_qubo_from_file(qubo_path)

    gt_combinations = get_ground_truth_combinations(disease_name, df)

    sorted_results = get_sorted_results_allowed_sizes(
        qubo, drug_ids, gt_combinations, allowed_sizes=allowed_sizes
    )

    predictions = [
        (combo, energy)
        for combo, energy, is_match in sorted_results
        if not is_match
    ]

    return predictions[:top_k]


def get_all_predictions(
    df: pd.DataFrame,
    DISEASE_CONFIGS: Dict[str, Dict[str, Any]],
    qubo_dir: str = "Results",
    top_k: int = 20,
    allowed_sizes: List[int] = [2, 3],
) -> Dict[str, List[Tuple[List[str], float]]]:
    """
    Returns dict mapping disease_name -> list of (drug_combination, energy) predictions.
    """
    all_predictions = {}
    for disease_name in DISEASE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {disease_name}")
        print(f"{'='*60}")
        preds = get_predictions_for_disease(
            disease_name, df, DISEASE_CONFIGS, qubo_dir, top_k, allowed_sizes
        )
        all_predictions[disease_name] = preds

        for rank, (combo, energy) in enumerate(preds, 1):
            print(f"  #{rank:2d}  E = {energy:12.4f}  {combo}")

    return all_predictions


def predictions_to_dataframe(
    all_predictions: Dict[str, List[Tuple[List[str], float]]],
) -> pd.DataFrame:
    """
    Convert the predictions dict into a flat DataFrame for export.
    """
    rows = []
    for disease, preds in all_predictions.items():
        for rank, (combo, energy) in enumerate(preds, 1):
            rows.append({
                'Disease': disease,
                'Rank': rank,
                'Energy': energy,
                'Combination': ' + '.join(combo),
                'Num Drugs': len(combo),
            })
    return pd.DataFrame(rows)


def get_drug_name(db_id, name_cache={}, manual_overrides={}):
    clean_id = db_id.strip().upper()

    if clean_id in manual_overrides:
        return manual_overrides[clean_id]
    
    if clean_id in name_cache:
        return name_cache[clean_id]
    
    url = f"https://mychem.info/v1/query?q=drugbank.id:{clean_id}&fields=chembl.pref_name"
    try:
        res = requests.get(url).json()
        if 'hits' in res and len(res['hits']) > 0:
            name = res['hits'][0].get('chembl', {}).get('pref_name', clean_id).title()
            name_cache[clean_id] = name
            time.sleep(0.1) 
            return name
    except:
        pass
    
    name_cache[clean_id] = f"{clean_id} (Unmapped)"
    return name_cache[clean_id]

def check_pubmed_synergy(drug1, drug2, disease):

    if "(Experimental)" in drug1 or "(Experimental)" in drug2:
        return 0
        
    query = f'("{drug1}" AND "{drug2}" AND "{disease}") AND ("synergy" OR "synergistic" OR "combination therapy" OR "efficacy")'
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&format=json&resultType=lite"
    
    try:
        res = requests.get(url).json()
        return res.get('hitCount', 0)
    except:
        return 0


def check_pubmed_synergy_combo(drugs: List[str], disease: str) -> int:
    """
    Generalization of check_pubmed_synergy to combinations of any size.
    Returns the Europe PMC hit count for the synergy-keyword query.
    """
    if any("(Experimental)" in d for d in drugs):
        return 0

    drug_clauses = " AND ".join(f'"{d}"' for d in drugs)
    query = (
        f'({drug_clauses} AND "{disease}") AND '
        f'("synergy" OR "synergistic" OR "combination therapy" OR "efficacy")'
    )
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={query}&format=json&resultType=lite"
    )
    try:
        return requests.get(url).json().get("hitCount", 0)
    except Exception:
        return 0


def validate_predictions(
    pred_df: pd.DataFrame,
    manual_overrides: Optional[Dict[str, str]] = None,
    sleep: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For each row in pred_df (Disease, Rank, Combination), look up drug names and query Europe PMC for synergy hits. 
    Returns a DataFrame with columns:
    Disease, Rank, Drugs, PubMed Hits.
    """
    if manual_overrides is None:
        manual_overrides = {}
    name_cache: Dict[str, str] = {}

    rows = []
    if verbose:
        print("Starting systematic validation...\n" + "-" * 50)

    for _, row in pred_df.iterrows():
        disease = row["Disease"]
        ids = row["Combination"].split(" + ")
        names = [
            get_drug_name(d, name_cache=name_cache, manual_overrides=manual_overrides)
            for d in ids
        ]
        hits = check_pubmed_synergy_combo(names, disease)

        if verbose:
            print(f"[{disease}] Rank {row['Rank']}: {' + '.join(names)}")
            print(f"   -> Validated in {hits} published papers.\n")

        rows.append({
            "Disease": disease,
            "Rank": int(row["Rank"]),
            "Drugs": " + ".join(names),
            "PubMed Hits": hits,
        })
        time.sleep(sleep)

    return pd.DataFrame(rows)


def save_validation_table_pdf(
    validated_df: pd.DataFrame,
    save_path: str,
    highlight_color: str = "#e8f5e9",
    header_color: str = "#40466e",
):
    cols = ["Rank", "Drugs", "PubMed Hits"]

    with PdfPages(save_path) as pdf:
        for disease in validated_df["Disease"].unique():
            sub = validated_df[validated_df["Disease"] == disease][cols].reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(10, 0.35 * len(sub) + 1.6))
            ax.axis("off")
            ax.set_title(disease, fontsize=14, fontweight="bold", pad=14)

            tbl = ax.table(
                cellText=sub.values,
                colLabels=sub.columns,
                cellLoc="center",
                loc="center",
                colWidths=[0.10, 0.65, 0.20],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.4)

            for j in range(len(cols)):
                tbl[(0, j)].set_facecolor(header_color)
                tbl[(0, j)].set_text_props(color="white", fontweight="bold")

            for i in range(len(sub)):
                if sub.iloc[i]["PubMed Hits"] > 0:
                    for j in range(len(cols)):
                        tbl[(i + 1, j)].set_facecolor(highlight_color)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

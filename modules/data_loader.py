from pathlib import Path
import pandas as pd

# Fichiers par défaut pour chaque module
DEFAULT_FILES = {
    "donnees": Path(__file__).resolve().parent.parent / "data" / "BASE_Données_projet_MIRACE-13_01-2025.xlsx",
    "profilage1": Path(__file__).resolve().parent.parent / "data" / "data_variable_revenuHt_Inflation.xlsx",
    "profilage2": Path(__file__).resolve().parent.parent / "data" / "data_facteurs_variables_resilience_covid.xlsx",
    "calculateur": Path(__file__).resolve().parent.parent / "data" / "sample_pays_avant_covid.xlsx",
}

def load_dataset(uploaded_file=None, mode="donnees"):
    """
    Retourne (dict[sheet -> DataFrame], chemin_utilisé)
    Si aucun fichier n'est uploadé, utilise un fichier par défaut en fonction du mode.
    """
    path = uploaded_file or DEFAULT_FILES.get(mode)

    if not path:
        raise ValueError(f"Mode inconnu ou fichier par défaut manquant pour le mode '{mode}'")

    if str(path).lower().endswith(".csv"):
        return {"CSV": pd.read_csv(path)}, path
    else:
        return pd.read_excel(path, sheet_name=None, engine="openpyxl"), path
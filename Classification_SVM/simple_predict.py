from pathlib import Path
import pandas as pd
#from SVM_models import SVM_basique as modele
#from SVM_models import SVM_E2 as modele
#from SVM_models import SVM_E3 as modele
from SVM_models import SVM_E4 as modele
#from SVM_models import SVM_E5 as modele
#from SVM_models import SVM_E6 as modele
#from SVM_models import SVM_E7 as modele
#from SVM_models import SVM_E9 as modele
#from SVM_models import SVM_E10 as modele
#from SVM_models import SVM_E11 as modele
#from SVM_models import SVM_E12 as modele
#from SVM_models import SVM_E13 as modele
#from SVM_models import SVM_E14 as modele
#from SVM_models import SVM_E15 as modele


from pathlib import Path
import os

# === OBLIGATOIRE : dossier local des données (hors repo) ===
DATA_DIR = Path(r"c:\Users\smnle\Desktop\M2\Apprentissage supervisée\Data challenge\Classification\data")
if not DATA_DIR:
    raise RuntimeError(
        "Définissez la variable DATA_DIR vers le dossier des données.\n"
    )

# Fichiers attendus dans ce dossier
TRAIN_CSV = DATA_DIR / "train_data.csv"
TEST_CSV  = DATA_DIR / "test_data.csv"
OUT_CSV   = DATA_DIR / "submission.csv"


def write_submission(model_module,
                     model_path: str,
                     input_csv: str,
                     output_csv: str,
                     id_col: str = "row_id",
                     target_col: str = "reservation_status"):
    """
    - Lit input_csv (doit contenir id_col)
    - drop id_col et target_col des features si présents
    - model_module.load(model_path) puis model_module.predict(X)
    - écrit output_csv avec colonnes: id_col, target_col (entiers)
    """
    input_path = Path(input_csv).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier test introuvable: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Fichier d'entraînement introuvable: {model_path}")

    df = pd.read_csv(input_path)
    if id_col not in df.columns:
        raise ValueError(f"Colonne identifiant '{id_col}' absente de {input_path}")

    # Features = toutes les colonnes sauf id et (éventuelle) target
    drop_cols = [c for c in (id_col, target_col) if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Charge et prédit
    model_module.load(str(model_path))
    preds = model_module.predict(X)

    # Écriture du fichier de soumission
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({id_col: df[id_col], target_col: pd.Series(preds).astype(int)})
    out.to_csv(output_path, index=False)
    return str(output_path)


if __name__ == "__main__":
    # BASE = dossier où se trouve ce script
    BASE = Path(__file__).resolve().parent

    write_submission(
        modele,
        model_path=str(TRAIN_CSV),       
        input_csv=str(TEST_CSV),
        output_csv=str(OUT_CSV),
        id_col="row_id",
        target_col="reservation_status"
    )
    print("Soumission écrite dans:", BASE / "data" / "submission.csv")

r"""Entraîne les modèles finaux sur TOUTES les séquences RoAD et les sauvegarde
pour l'interface (app_road.py).

Lancer :  python train_final_models.py      (même dossier que run_road_final.py)
Sorties : models_final/road_models.joblib   et   demo_sequences/*.csv
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from run_road_final import (Dataset, QUANTILES, SCENARIOS, SEED, STATS, WINDOW,
                            build_records, pca_error, sensor_names)

OUT = Path("models_final")
DEMO = Path("demo_sequences")
MAX_DEMO_ROWS = 6000          # 600 s à 10 Hz


def choose_quantile(records):
    """Choisit le quantile du seuil PCA sur une validation par séquence."""
    idx = np.arange(len(records))
    fit_i, val_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(idx, groups=idx))
    X_fit = np.concatenate([records[i]["X"] for i in fit_i]); y_fit = np.concatenate([records[i]["y"] for i in fit_i])
    X_val = np.concatenate([records[i]["X"] for i in val_i]); y_val = np.concatenate([records[i]["y"] for i in val_i])
    if y_val.sum() == 0 or (y_fit == 0).sum() == 0:
        return 0.95
    sc = StandardScaler().fit(X_fit[y_fit == 0])
    pca = PCA(n_components=0.95, random_state=SEED).fit(sc.transform(X_fit[y_fit == 0]))
    err_fit, err_val = pca_error(sc, pca, X_fit[y_fit == 0]), pca_error(sc, pca, X_val)
    best_q, best_f1 = 0.95, -1.0
    for q in QUANTILES:
        f = f1_score(y_val, (err_val > np.quantile(err_fit, q)).astype(int), zero_division=0)
        if f > best_f1:
            best_q, best_f1 = q, f
    return best_q


def export_demo(road):
    DEMO.mkdir(exist_ok=True)
    for sc in ["training"] + SCENARIOS:
        for i, rec in enumerate(road.sets[sc][:2]):
            rec = np.asarray(rec)
            start = 0
            if rec.shape[1] > 86 and (rec[:, 86] > 0).any():
                start = max(0, int(np.argmax(rec[:, 86] > 0)) - 2000)
            part = rec[start:start + MAX_DEMO_ROWS, :87]
            np.savetxt(DEMO / f"{sc}_{i}.csv", part, delimiter=",", fmt="%.5g")
    print("Séquences de démonstration :", sorted(p.name for p in DEMO.glob("*.csv")))


def main():
    OUT.mkdir(exist_ok=True)
    road = Dataset(normalize=True)
    records = build_records(road)
    X = np.concatenate([r["X"] for r in records]); y = np.concatenate([r["y"] for r in records])
    print(f"{len(records)} séquences, {len(y)} fenêtres, {int(y.sum())} anormales")

    q = choose_quantile(records)
    scaler = StandardScaler().fit(X[y == 0])
    pca = PCA(n_components=0.95, random_state=SEED).fit(scaler.transform(X[y == 0]))
    threshold = float(np.quantile(pca_error(scaler, pca, X[y == 0]), q))

    tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=3, class_weight="balanced",
                                  random_state=SEED).fit(X, y)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                                random_state=SEED, n_jobs=-1).fit(X, y)
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, class_weight="balanced",
                                              random_state=SEED)).fit(X, y)

    bundle = {"RandomForest": rf, "SVM": svm, "DecisionTree": tree,
              "scaler": scaler, "pca": pca, "threshold": threshold, "quantile": q,
              "window": WINDOW, "stats": STATS, "sensor_names": sensor_names(road)}
    joblib.dump(bundle, OUT / "road_models.joblib")
    print(f"Modèles sauvegardés : {(OUT / 'road_models.joblib').resolve()}")
    print(f"Seuil PCA : quantile {q}, valeur {threshold:.4f}")
    export_demo(road)


if __name__ == "__main__":
    main()

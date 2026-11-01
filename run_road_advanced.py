r"""RoAD KUKA — expérience avancée pour le projet Souhaieb/Salsabil.

Améliorations incluses :
1. fenêtres de 5, 10 et 20 secondes (50, 100, 200 points à 10 Hz) ;
2. indicateurs temporels : moyenne, écart-type, RMS, min, max, crête-à-crête,
   énergie moyenne et kurtosis calculée sans dépendance scipy ;
3. évaluation séparée des scénarios collision, weight et velocity ;
4. comparaison SVM, arbre, Isolation Forest et PCA ;
5. recherche du seuil PCA selon F1 et rappel ;
6. Health Index basé sur le rapport erreur PCA / seuil ;
7. export des résultats en CSV et des modèles en joblib.

Lancer dans PowerShell :
    python run_road_advanced.py
"""

from pathlib import Path
import csv
import json
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from RoADDataset import Dataset

N_FEATURES = 86
WINDOWS = {50: "5s", 100: "10s", 200: "20s"}
SCENARIOS = ["collision", "control", "weight", "velocity"]


def kurtosis_axis1(x):
    """Kurtosis de Pearson par fenêtre et par variable."""
    mean = x.mean(axis=1, keepdims=True)
    centered = x - mean
    variance = np.mean(centered ** 2, axis=1)
    fourth = np.mean(centered ** 4, axis=1)
    return fourth / (variance ** 2 + 1e-8)


def window_features(recording, window):
    """Calcule les indicateurs par fenêtre et le label binaire de la fenêtre."""
    signals = recording[:, :N_FEATURES].astype(np.float32)
    # Colonne 0 : ID de l'action ; les 85 autres colonnes sont les capteurs.
    signals = signals[:, 1:]
    labels = (
        recording[:, N_FEATURES].astype(int) > 0
        if recording.shape[1] > N_FEATURES
        else np.zeros(recording.shape[0], dtype=int)
    )
    n_windows = len(signals) // window
    if n_windows == 0:
        return np.empty((0, 85 * 8), dtype=np.float32), np.empty(0, dtype=int)
    x = signals[: n_windows * window].reshape(n_windows, window, -1)
    y = labels[: n_windows * window].reshape(n_windows, window)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    minimum = x.min(axis=1)
    maximum = x.max(axis=1)
    peak_to_peak = maximum - minimum
    energy = np.mean(x ** 2, axis=1)
    kurtosis = kurtosis_axis1(x)
    features = np.concatenate(
        [mean, std, rms, minimum, maximum, peak_to_peak, energy, kurtosis], axis=1
    )
    window_labels = (y.max(axis=1) > 0).astype(int)
    return features.astype(np.float32), window_labels


def make_windows(recordings, window):
    xs, ys = [], []
    for recording in recordings:
        x, y = window_features(recording, window)
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        return np.empty((0, 85 * 8), dtype=np.float32), np.empty(0, dtype=int)
    return np.concatenate(xs), np.concatenate(ys)


def split_recordings(dataset, names, train_fraction=0.70):
    """Sépare des séquences complètes ; une séquence ne traverse pas train/test."""
    train, test = [], []
    for name in names:
        values = dataset.sets[name]
        if len(values) <= 1:
            test.extend(values)
            continue
        cut = max(1, min(len(values) - 1, int(len(values) * train_fraction)))
        train.extend(values[:cut])
        test.extend(values[cut:])
    return train, test


def metrics(y_true, pred):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision_anomaly": precision_score(y_true, pred, zero_division=0),
        "recall_anomaly": recall_score(y_true, pred, zero_division=0),
        "f1_anomaly": f1_score(y_true, pred, zero_division=0),
    }


def save_row(rows, window, model, scenario, values):
    rows.append({"window": window, "model": model, "scenario": scenario, **values})


def evaluate_model(rows, window, model_name, y_test, pred, scenario_ids):
    save_row(rows, window, model_name, "all_test", metrics(y_test, pred))
    for scenario in SCENARIOS:
        mask = scenario_ids == scenario
        if mask.any() and len(np.unique(y_test[mask])) > 1:
            save_row(rows, window, model_name, scenario, metrics(y_test[mask], pred[mask]))
        elif mask.any():
            # Métriques descriptives même lorsqu'un scénario contient une seule classe.
            save_row(rows, window, model_name, scenario, metrics(y_test[mask], pred[mask]))


def main():
    road = Dataset(normalize=True)
    labeled_train, labeled_test = split_recordings(road, SCENARIOS)
    # La liste des scénarios test doit garder l'identité de chaque séquence.
    test_recordings = []
    for name in SCENARIOS:
        values = road.sets[name]
        if len(values) <= 1:
            selected = values
        else:
            cut = max(1, min(len(values) - 1, int(len(values) * 0.70)))
            selected = values[cut:]
        test_recordings.extend((name, recording) for recording in selected)

    output = Path("advanced_results")
    output.mkdir(exist_ok=True)
    all_rows = []

    for window, duration in WINDOWS.items():
        train_recordings = list(road.sets["training"]) + labeled_train
        X_train, y_train = make_windows(train_recordings, window)
        test_x, test_y, scenario_ids = [], [], []
        for scenario, recording in test_recordings:
            x_part, y_part = make_windows([recording], window)
            test_x.append(x_part)
            test_y.append(y_part)
            scenario_ids.extend([scenario] * len(y_part))
        X_test = np.concatenate(test_x)
        y_test = np.concatenate(test_y)
        scenario_ids = np.array(scenario_ids)
        print(f"\n========== Fenêtre {duration} ({window} points) ==========")
        print("Train/test :", X_train.shape, X_test.shape)

        svm = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced", C=10, random_state=42),
        )
        svm.fit(X_train, y_train)
        evaluate_model(all_rows, window, "SVM", y_test, svm.predict(X_test), scenario_ids)

        tree = DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42
        )
        tree.fit(X_train, y_train)
        evaluate_model(all_rows, window, "DecisionTree", y_test, tree.predict(X_test), scenario_ids)

        normal_train, _ = make_windows(road.sets["training"], window)
        isolation = IsolationForest(
            n_estimators=200, contamination="auto", random_state=42, n_jobs=-1
        )
        isolation.fit(normal_train)
        pred_iso = (isolation.predict(X_test) == -1).astype(int)
        evaluate_model(all_rows, window, "IsolationForest", y_test, pred_iso, scenario_ids)

        scaler = StandardScaler()
        normal_scaled = scaler.fit_transform(normal_train)
        pca = PCA(n_components=0.95, random_state=42)
        normal_rec = pca.inverse_transform(pca.fit_transform(normal_scaled))
        train_error = np.mean((normal_scaled - normal_rec) ** 2, axis=1)
        test_scaled = scaler.transform(X_test)
        test_rec = pca.inverse_transform(pca.transform(test_scaled))
        test_error = np.mean((test_scaled - test_rec) ** 2, axis=1)

        # Recherche de seuils : meilleur F1 et meilleur rappel avec précision >= 0.70.
        threshold_rows = []
        for quantile in [0.90, 0.925, 0.95, 0.975, 0.99]:
            threshold = float(np.quantile(train_error, quantile))
            pred = (test_error > threshold).astype(int)
            values = metrics(y_test, pred)
            threshold_rows.append({"quantile": quantile, "threshold": threshold, **values})
            save_row(all_rows, window, f"PCA_q{quantile}", "all_test", values)
        best_f1 = max(threshold_rows, key=lambda r: r["f1_anomaly"])
        candidates = [r for r in threshold_rows if r["precision_anomaly"] >= 0.70]
        best_recall = max(candidates or threshold_rows, key=lambda r: r["recall_anomaly"])
        selected = best_f1
        pred_pca = (test_error > selected["threshold"]).astype(int)
        evaluate_model(all_rows, window, "PCA_selected", y_test, pred_pca, scenario_ids)

        # Health Index : 0 normal, >1 au-dessus du seuil PCA.
        health_index = np.clip(test_error / selected["threshold"], 0, 3)
        np.save(output / f"health_index_{window}.npy", health_index)
        with open(output / f"pca_thresholds_{window}.json", "w", encoding="utf-8") as f:
            json.dump({"all_thresholds": threshold_rows, "selected": selected, "best_recall": best_recall}, f, indent=2)
        joblib.dump(
            {"scaler": scaler, "pca": pca, "threshold": selected["threshold"], "window": window},
            output / f"pca_{window}.joblib",
        )
        print("PCA composantes :", pca.n_components_)
        print("Seuil sélectionné :", round(selected["threshold"], 6))
        print("F1 PCA sélectionnée :", round(selected["f1_anomaly"], 4))
        print("Rappel PCA sélectionnée :", round(selected["recall_anomaly"], 4))

    with open(output / "advanced_metrics.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["window", "model", "scenario", "accuracy", "precision_anomaly", "recall_anomaly", "f1_anomaly"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nRésultats exportés dans :", output.resolve())
    print("Fichier principal :", (output / "advanced_metrics.csv").resolve())


if __name__ == "__main__":
    main()

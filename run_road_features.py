r"""RoAD KUKA - deuxième expérience avec fenêtres temporelles.

À exécuter dans le même environnement que run_road_models.py :
    python run_road_features.py

Le script regroupe 100 échantillons (10 secondes après le rééchantillonnage à
10 Hz) et calcule moyenne, écart-type, RMS, minimum et maximum pour les
variables capteurs. Les résultats sont donc des indicateurs par fenêtre, pas
des décisions sur une mesure isolée.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from RoADDataset import Dataset

N_FEATURES = 86
WINDOW = 100  # 100 échantillons à 10 Hz = 10 secondes


def window_features(recording):
    """Retourne une ligne d'indicateurs par fenêtre et son label binaire."""
    signals = recording[:, :N_FEATURES].astype(np.float32)
    # L'action ID est catégorielle : on ne calcule pas de statistiques dessus.
    signals = signals[:, 1:]
    if recording.shape[1] > N_FEATURES:
        labels = (recording[:, N_FEATURES] > 0).astype(int)
    else:
        labels = np.zeros(recording.shape[0], dtype=int)

    n_windows = len(signals) // WINDOW
    if n_windows == 0:
        return np.empty((0, (N_FEATURES - 1) * 5), dtype=np.float32), np.empty(0, dtype=int)

    signals = signals[:n_windows * WINDOW].reshape(n_windows, WINDOW, -1)
    labels = labels[:n_windows * WINDOW].reshape(n_windows, WINDOW)
    mean = signals.mean(axis=1)
    std = signals.std(axis=1)
    rms = np.sqrt(np.mean(signals ** 2, axis=1))
    minimum = signals.min(axis=1)
    maximum = signals.max(axis=1)
    features = np.concatenate([mean, std, rms, minimum, maximum], axis=1)
    window_labels = (labels.max(axis=1) > 0).astype(int)
    return features, window_labels


def make_windows(recordings):
    xs, ys = [], []
    for recording in recordings:
        x, y = window_features(recording)
        if len(x):
            xs.append(x)
            ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def split_recordings(dataset, names, train_fraction=0.70):
    train, test = [], []
    for name in names:
        values = dataset.sets[name]
        cut = max(1, int(len(values) * train_fraction))
        train.extend(values[:cut])
        test.extend(values[cut:])
    return train, test


def evaluate(name, y_true, pred):
    print(f"\n===== {name} =====")
    print("F1-score anomalie :", round(f1_score(y_true, pred, zero_division=0), 4))
    print("Matrice de confusion [normal, anomalie] :")
    print(confusion_matrix(y_true, pred, labels=[0, 1]))
    print(classification_report(y_true, pred, target_names=["normal", "anomalie"], zero_division=0))


def main():
    road = Dataset(normalize=True)
    scenario_names = ["collision", "control", "weight", "velocity"]
    labeled_train, labeled_test = split_recordings(road, scenario_names)

    # Apprentissage supervisé : training normal + partie annotée des scénarios.
    X_train, y_train = make_windows(list(road.sets["training"]) + labeled_train)
    X_test, y_test = make_windows(labeled_test)
    print("Fenêtre :", WINDOW, "échantillons = 10 secondes")
    print("X_train / X_test :", X_train.shape, X_test.shape)
    print("Labels train :", np.unique(y_train, return_counts=True))
    print("Labels test  :", np.unique(y_test, return_counts=True))

    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
    )
    svm.fit(X_train, y_train)
    evaluate("SVM + indicateurs temporels", y_test, svm.predict(X_test))

    tree = DecisionTreeClassifier(
        max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42
    )
    tree.fit(X_train, y_train)
    evaluate("Arbre + indicateurs temporels", y_test, tree.predict(X_test))

    # Méthodes non supervisées : apprentissage exclusivement sur les fenêtres normales.
    X_normal, _ = make_windows(road.sets["training"])

    isolation = IsolationForest(
        n_estimators=200, contamination="auto", random_state=42, n_jobs=-1
    )
    isolation.fit(X_normal)
    pred_iso = (isolation.predict(X_test) == -1).astype(int)
    evaluate("Isolation Forest + indicateurs", y_test, pred_iso)

    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(X_normal)
    pca = PCA(n_components=0.95, random_state=42)
    normal_rec = pca.inverse_transform(pca.fit_transform(normal_scaled))
    train_error = np.mean((normal_scaled - normal_rec) ** 2, axis=1)
    threshold = np.quantile(train_error, 0.95)
    test_scaled = scaler.transform(X_test)
    test_rec = pca.inverse_transform(pca.transform(test_scaled))
    test_error = np.mean((test_scaled - test_rec) ** 2, axis=1)
    evaluate("PCA + indicateurs temporels", y_test, (test_error > threshold).astype(int))
    print("Composantes PCA :", pca.n_components_)
    print("Seuil PCA :", round(float(threshold), 6))

    output = Path("models_features")
    output.mkdir(exist_ok=True)
    joblib.dump(svm, output / "svm_features.joblib")
    joblib.dump(tree, output / "tree_features.joblib")
    joblib.dump(isolation, output / "isolation_features.joblib")
    joblib.dump({"scaler": scaler, "pca": pca, "threshold": threshold}, output / "pca_features.joblib")
    print("Modèles sauvegardés dans :", output.resolve())


if __name__ == "__main__":
    main()

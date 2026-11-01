r"""Projet KUKA LBR iiwa / RoAD
Chargement du dataset et comparaison de modèles de détection d'anomalies.

Installation :
    python -m venv .venv
    source .venv/bin/activate          # Linux/macOS
    .venv\Scripts\activate             # Windows PowerShell
    pip install git+https://gitlab.com/AlessioMascolini/roaddataset/
    pip install numpy scikit-learn pandas joblib

Exécution :
    python run_road_models.py
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


# Les 86 premières colonnes sont les variables capteurs.
# La 87e colonne est le label : 0 = normal, 1 ou 2 = anomalie(s).
N_FEATURES = 86


def concatenate_recordings(recordings):
    """Transforme une liste de séquences en une seule matrice X et un vecteur y."""
    # On traite chaque séquence avant la concaténation, car training contient
    # 86 colonnes alors que les autres sous-ensembles peuvent en contenir 87.
    # Les 86 premières colonnes sont toujours les variables capteurs.
    feature_blocks = []
    label_blocks = []
    for recording in recordings:
        X_part = recording[:, :N_FEATURES].astype(np.float32)
        if recording.shape[1] > N_FEATURES:
            y_part = recording[:, N_FEATURES].astype(int)
        else:
            # training = fonctionnement normal sans colonne de label publiée.
            y_part = np.zeros(recording.shape[0], dtype=int)
        feature_blocks.append(X_part)
        label_blocks.append(y_part)
    X = np.concatenate(feature_blocks, axis=0)
    y = np.concatenate(label_blocks, axis=0)
    return X, y


def split_by_recording(dataset, names, train_fraction=0.70):
    """Sépare des séquences complètes, pas des lignes aléatoires."""
    train_recordings, test_recordings = [], []
    for name in names:
        recordings = dataset.sets[name]
        cut = max(1, int(len(recordings) * train_fraction))
        train_recordings.extend(recordings[:cut])
        test_recordings.extend(recordings[cut:])
    return train_recordings, test_recordings


def evaluate(name, y_true, y_pred):
    """Affiche les résultats binaires : normal contre anomalie."""
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    print(f"\n===== {name} =====")
    print("F1-score anomalie :", round(f1_score(y_true_binary, y_pred_binary, zero_division=0), 4))
    print("Matrice de confusion [normal, anomalie] :")
    print(confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]))
    print(classification_report(
        y_true_binary,
        y_pred_binary,
        target_names=["normal", "anomalie"],
        zero_division=0,
    ))


def main():
    print("Chargement de RoAD...")
    # normalize=True est le comportement recommandé du package :
    # la normalisation est ajustée sur le sous-ensemble training.
    road = Dataset(normalize=True)
    print("Sous-ensembles :", list(road.sets.keys()))
    print("Colonnes disponibles :", len(road.columns))
    print("Nombre d'enregistrements :")
    for name, recordings in road.sets.items():
        print(f"  {name:10s}: {len(recordings)}")

    # Séparation par enregistrement pour éviter qu'une même séquence apparaisse
    # à la fois dans l'apprentissage et dans le test.
    # Le sous-ensemble training sert de référence normale. Pour les modèles
    # supervisés, on ajoute 70 % des séquences annotées aux données d'apprentissage.
    scenario_names = ["collision", "control", "weight", "velocity"]
    labeled_train, labeled_test = split_by_recording(road, scenario_names)
    train_recordings = list(road.sets["training"]) + labeled_train
    X_train, y_train = concatenate_recordings(train_recordings)
    X_test, y_test = concatenate_recordings(labeled_test)

    print("\nDimensions X_train / X_test :", X_train.shape, X_test.shape)
    print("Labels train :", np.unique(y_train, return_counts=True))
    print("Labels test  :", np.unique(y_test, return_counts=True))

    # 1) SVM supervisé : utilise les labels du training.
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
    )
    svm.fit(X_train, y_train)
    evaluate("SVM supervisé", y_test, svm.predict(X_test))

    # 2) Arbre de décision supervisé : facile à interpréter dans le rapport.
    tree = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X_train, y_train)
    evaluate("Arbre de décision", y_test, tree.predict(X_test))

    # Pour les méthodes non supervisées, on apprend uniquement le fonctionnement normal.
    # Pour l'apprentissage non supervisé, on ne garde que les observations
    # normales du sous-ensemble training, afin de ne pas apprendre les anomalies.
    X_training_only, y_training_only = concatenate_recordings(road.sets["training"])
    X_normal = X_training_only[y_training_only == 0]

    # 3) Isolation Forest : détecte les observations éloignées du comportement normal.
    isolation = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    isolation.fit(X_normal)
    # sklearn renvoie +1 normal et -1 anomalie.
    pred_iso = (isolation.predict(X_test) == -1).astype(int)
    evaluate("Isolation Forest", y_test, pred_iso)

    # 4) PCA : score de reconstruction ; seuil fixé au 95e percentile du normal.
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    pca = PCA(n_components=0.95, random_state=42)
    normal_reconstructed = pca.inverse_transform(pca.fit_transform(X_normal_scaled))
    train_error = np.mean((X_normal_scaled - normal_reconstructed) ** 2, axis=1)
    threshold = np.quantile(train_error, 0.95)

    X_test_scaled = scaler.transform(X_test)
    test_reconstructed = pca.inverse_transform(pca.transform(X_test_scaled))
    test_error = np.mean((X_test_scaled - test_reconstructed) ** 2, axis=1)
    pred_pca = (test_error > threshold).astype(int)
    print("Nombre de composantes PCA :", pca.n_components_)
    print("Seuil PCA :", round(float(threshold), 6))
    evaluate("PCA + erreur de reconstruction", y_test, pred_pca)

    # Sauvegarde des modèles pour une future prédiction sur de nouvelles séquences.
    output = Path("models_road")
    output.mkdir(exist_ok=True)
    joblib.dump(svm, output / "svm.joblib")
    joblib.dump(tree, output / "decision_tree.joblib")
    joblib.dump(isolation, output / "isolation_forest.joblib")
    joblib.dump({"scaler": scaler, "pca": pca, "threshold": threshold}, output / "pca.joblib")
    print(f"\nModèles sauvegardés dans : {output.resolve()}")


if __name__ == "__main__":
    main()

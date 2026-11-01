r"""Interface de surveillance - KUKA LBR iiwa / RoAD.

Lancer :  streamlit run app_road.py
Prérequis : python train_final_models.py (crée models_final/ et demo_sequences/),
            puis run_road_final.py et run_fmds_simule.py pour les onglets résultats.
Les séquences téléversées doivent être au même format normalisé que RoAD
(86 colonnes : action ID + 85 capteurs, + 1 colonne label optionnelle).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models_final" / "road_models.joblib"
DEMO_DIR = BASE / "demo_sequences"
RES_DIR = BASE / "final_results"
FMDS_DIR = BASE / "fmds_results"
WINDOW, WINDOW_S, ALPHA = 100, 10, 0.3
N_FEATURES, N_SENSORS = 86, 85

st.set_page_config(page_title="KUKA iiwa - maintenance conditionnelle", layout="wide")


# ------------------------------------------------------------------ calculs
def kurtosis_axis1(x):
    c = x - x.mean(axis=1, keepdims=True)
    return np.mean(c ** 4, axis=1) / (np.mean(c ** 2, axis=1) ** 2 + 1e-8)


def window_features(arr):
    sig = arr[:, :N_FEATURES].astype(np.float32)[:, 1:]
    lab = (arr[:, N_FEATURES] > 0).astype(int) if arr.shape[1] > N_FEATURES else None
    n = len(sig) // WINDOW
    if n == 0:
        return None, None
    x = sig[: n * WINDOW].reshape(n, WINDOW, -1)
    mx, mn = x.max(axis=1), x.min(axis=1)
    feats = np.concatenate([x.mean(axis=1), x.std(axis=1), np.sqrt(np.mean(x ** 2, axis=1)), mn, mx,
                            mx - mn, np.mean(x ** 2, axis=1), kurtosis_axis1(x)], axis=1)
    y = lab[: n * WINDOW].reshape(n, WINDOW).max(axis=1) if lab is not None else None
    return feats.astype(np.float32), y


def pca_error(scaler, pca, x):
    z = scaler.transform(x)
    return np.mean((z - pca.inverse_transform(pca.transform(z))) ** 2, axis=1)


def ewma(v, alpha=ALPHA):
    out, acc = np.empty(len(v)), v[0]
    for i, val in enumerate(v):
        acc = alpha * val + (1 - alpha) * acc
        out[i] = acc
    return out


def keep_runs(pred, k=2):
    """Garde les fenêtres signalées appartenant à une série d'au moins k fenêtres consécutives."""
    out, i, n = np.zeros_like(pred), 0, len(pred)
    while i < n:
        if pred[i]:
            j = i
            while j < n and pred[j]:
                j += 1
            if j - i >= k:
                out[i:j] = 1
            i = j
        else:
            i += 1
    return out


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def read_sequence(src):
    df = pd.read_csv(src, header=None)
    if pd.to_numeric(df.iloc[0], errors="coerce").isna().any():
        if hasattr(src, "seek"):
            src.seek(0)
        df = pd.read_csv(src, header=0)
    return df.apply(pd.to_numeric, errors="coerce").dropna().to_numpy()


def amdec_table():
    rows = [
        ("Sécurité", "Détection tardive ou manquée", "Seuil inadapté, rappel limité (~50 %)",
         "Anomalie non vue, risque équipement/opérateur", 10, 3, 7,
         "Seuil validé + inspection périodique en complément",
         "G=10 danger ; O=3 dépend du seuil ; D=7 car le rappel du modèle est ~50 %"),
        ("Mouvement", "Trajectoire / vitesse anormale", "Commande, perturbation",
         "Erreur de mouvement, cycle interrompu", 8, 5, 5,
         "Surveiller écart-type et crête-à-crête des IMU",
         "D=5 : détection de la vitesse variable selon le modèle (rappel 0,25 à 0,90)"),
        ("Articulation", "Vibration excessive", "Usure, jeu, désalignement",
         "Instabilité, perte de précision", 8, 5, 5,
         "Indicateur vibratoire robuste (RMS, écart-type)",
         "D=5 : les indicateurs IMU sont informatifs mais le rappel reste limité"),
        ("Capteur", "Mesure erronée ou perdue", "Bruit, perte de connexion",
         "Fausse alarme ou défaut non détecté", 7, 4, 7,
         "Contrôle de cohérence, redondance des capteurs",
         "D=7 : un capteur défaillant peut imiter ou masquer une anomalie"),
        ("Structure", "Collision", "Obstacle, choc",
         "Désalignement, dommage", 9, 3, 7,
         "Alerte RF/SVM puis inspection après chaque alarme",
         "D=7 : collision = scénario le moins bien détecté (rappel ~0,4-0,5 pour RF/SVM)"),
        ("Actionneur", "Consommation anormale", "Surcharge, frottement",
         "Échauffement, arrêt de sécurité", 9, 5, 4,
         "Suivre la puissance du robot",
         "D=4 : la puissance est le capteur le plus important du modèle"),
        ("Charge / préhension", "Charge inhabituelle (weight)", "Mauvaise charge",
         "Effort excessif", 8, 4, 4,
         "Vérifier la charge avant cycle",
         "D=4 : weight est le scénario le mieux détecté"),
        ("Programme", "Séquence de tâche incorrecte", "Erreur de programme",
         "Cycle incorrect", 7, 3, 6,
         "Comparer à la séquence de référence",
         "D=6 : non évalué spécifiquement, valeur prudente"),
    ]
    df = pd.DataFrame(rows, columns=["Fonction", "Mode de défaillance", "Cause possible", "Effet observable",
                                     "G", "O", "D", "Action", "Justification des notes"])
    df.insert(7, "NPR", df["G"] * df["O"] * df["D"])
    return df.sort_values("NPR", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------- pages
st.title("KUKA LBR iiwa - détection d'anomalies et maintenance conditionnelle")
st.caption("Projet FMDS / maintenance prédictive - jeu de données RoAD")
bundle = load_bundle()
tab_diag, tab_res, tab_fmds, tab_amdec, tab_rule = st.tabs(
    ["Diagnostic d'une séquence", "Résultats des modèles", "FMDS (simulé)", "AMDEC", "Règle de maintenance"])

with tab_diag:
    if bundle is None:
        st.error("Modèles introuvables. Lancez d'abord : python train_final_models.py")
    else:
        c1, c2, c3 = st.columns(3)
        source = c1.radio("Source", ["Séquence de démonstration", "Téléverser un CSV"])
        model_name = c2.selectbox("Modèle d'alarme", ["RandomForest", "SVM", "DecisionTree"])
        alarm_pct = c3.slider("Seuil d'alarme (% de fenêtres signalées)", 1, 50, 5)
        consec = st.checkbox("Exiger 2 fenêtres consécutives signalées (filtre les fausses alarmes isolées)")
        arr, name = None, ""
        if source == "Séquence de démonstration":
            demos = sorted(DEMO_DIR.glob("*.csv")) if DEMO_DIR.exists() else []
            if not demos:
                st.warning("Aucune séquence dans demo_sequences/. Lancez train_final_models.py.")
            else:
                choice = st.selectbox("Séquence", [d.name for d in demos])
                arr, name = read_sequence(DEMO_DIR / choice), choice
        else:
            up = st.file_uploader("Séquence CSV (86 ou 87 colonnes, déjà normalisée)", type=["csv"])
            if up is not None:
                arr, name = read_sequence(up), up.name

        if arr is not None:
            if arr.shape[1] not in (86, 87):
                st.error(f"{arr.shape[1]} colonnes trouvées ; il en faut 86 ou 87.")
            else:
                X, y_true = window_features(arr)
                if X is None:
                    st.error(f"Séquence trop courte (minimum {WINDOW} lignes, soit {WINDOW_S} s).")
                else:
                    pred = bundle[model_name].predict(X).astype(int)
                    if consec:
                        pred = keep_runs(pred)
                    hi = pca_error(bundle["scaler"], bundle["pca"], X) / bundle["threshold"]
                    smooth = ewma(hi)
                    share = pred.mean() * 100
                    if pred.sum() == 0:
                        st.success(f"NORMAL - aucune fenêtre signalée par {model_name}.")
                    elif share >= alarm_pct:
                        st.error(f"ALARME - {share:.0f} % des fenêtres signalées par {model_name} : inspection recommandée.")
                    else:
                        st.warning(f"SURVEILLANCE - {share:.0f} % des fenêtres signalées par {model_name} (sous le seuil d'alarme).")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Fenêtres analysées (10 s)", len(pred))
                    m2.metric("Fenêtres signalées", int(pred.sum()))
                    m3.metric("Health Index moyen (PCA)", f"{hi.mean():.2f}")

                    t = np.arange(len(hi)) * WINDOW_S
                    fig, ax = plt.subplots(figsize=(10, 3.4))
                    ax.plot(t, hi, lw=0.8, alpha=0.5, label="Health Index (PCA)")
                    ax.plot(t, smooth, lw=1.8, label="Health Index lissé (EWMA)")
                    ax.axhline(1, color="r", ls="--", lw=1, label="seuil PCA = 1")
                    top = max(hi.max(), 1.2) * 1.05
                    if y_true is not None:
                        ax.fill_between(t, 0, top, where=y_true > 0, color="orange", alpha=0.2, label="anomalie annotée")
                    ax.scatter(t[pred > 0], np.full(int(pred.sum()), top * 0.97), marker="v", color="red",
                               s=18, label=f"signalé par {model_name}")
                    ax.set_xlabel("temps (s)"); ax.set_ylabel("Health Index"); ax.legend(fontsize=7, ncol=3)
                    st.pyplot(fig); plt.close(fig)
                    st.caption("Le Health Index PCA est un indicateur de tendance (écart au fonctionnement normal), "
                               "pas un détecteur fiable : il produit beaucoup de fausses alarmes.")

                    if y_true is not None:
                        st.subheader("Comparaison avec les anomalies annotées")
                        cm = confusion_matrix(y_true, pred, labels=[0, 1])
                        st.table(pd.DataFrame(cm, index=["réel normal", "réel anomalie"], columns=["prédit normal", "prédit anomalie"]))
                        if y_true.sum() > 0:
                            st.write(f"Précision {precision_score(y_true, pred, zero_division=0):.2f} - "
                                     f"Rappel {recall_score(y_true, pred, zero_division=0):.2f} - "
                                     f"F1 {f1_score(y_true, pred, zero_division=0):.2f}")
                        else:
                            st.write(f"Aucune anomalie annotée : taux de fausses alarmes = {pred.mean():.2f}")
                        if source == "Séquence de démonstration":
                            st.caption("Démonstration sur une séquence vue à l'entraînement (modèles entraînés sur les 15 séquences) : "
                                       "ces scores ne mesurent pas la performance réelle. Voir l'onglet « Résultats des modèles » "
                                       "(validation croisée par séquence).")
                    out = pd.DataFrame({"temps_s": t, "signale": pred, "health_index": hi, "health_index_ewma": smooth})
                    st.download_button("Télécharger les prédictions (CSV)", out.to_csv(index=False), f"predictions_{name}")

with tab_res:
    st.subheader("Validation croisée par séquence (moyenne ± écart-type sur les folds)")
    f = RES_DIR / "metrics_summary.csv"
    if f.exists():
        st.dataframe(pd.read_csv(f).round(3), use_container_width=True)
    else:
        st.info("Lancez run_road_final.py pour générer final_results/.")
    f = RES_DIR / "metrics_per_scenario.csv"
    if f.exists():
        st.subheader("Par scénario (rappel ; fausses alarmes si aucune anomalie)")
        st.dataframe(pd.read_csv(f).round(3), use_container_width=True)
    for img, cap in (("confusion_matrices.png", "Matrices de confusion cumulées"),
                     ("importance_sensors.png", "Capteurs les plus informatifs (Random Forest)"),
                     ("health_index_examples.png", "Exemples de Health Index")):
        if (RES_DIR / img).exists():
            st.image(str(RES_DIR / img), caption=cap)
    st.warning("Limites : très peu de séquences annotées (résultats fragiles), rappel d'environ 50 % pour RF/SVM, "
               "pas de RUL possible (RoAD ne contient pas de dégradation jusqu'à la panne).")

with tab_fmds:
    st.error("DONNÉES SIMULÉES - ce bloc ne vient pas de RoAD. Les temps de défaillance suivent une loi de Weibull "
             "simulée (β = 2,5 ; η = 1500 h ; censure à 1200 h) et les temps de réparation sont supposés.")
    f = FMDS_DIR / "indicateurs_fmds.csv"
    if f.exists():
        st.dataframe(pd.read_csv(f), use_container_width=True)
        for fn, cap in (("estimateurs.csv", "Estimateurs de la loi de Weibull"),
                        ("lois_aic.csv", "Comparaison des lois (AIC, plus petit = meilleur)")):
            if (FMDS_DIR / fn).exists():
                st.subheader(cap)
                st.dataframe(pd.read_csv(FMDS_DIR / fn).round(3), use_container_width=True)
        for img in ("fmds_courbes.png", "fmds_estimateurs.png"):
            if (FMDS_DIR / img).exists():
                st.image(str(FMDS_DIR / img))
    else:
        st.info("Lancez run_fmds_simule.py pour générer fmds_results/.")

with tab_amdec:
    st.write("NPR = G × O × D. Notes = jugements d'ingénieur à justifier (aucun historique de pannes dans le jeu de données).")
    seuil = st.slider("Seuil d'action prioritaire (NPR ≥)", 50, 300, 180)
    am = amdec_table()
    st.dataframe(am.style.apply(lambda r: ["background-color: #ffd9d9" if r["NPR"] >= seuil else "" for _ in r], axis=1),
                 use_container_width=True)
    st.download_button("Télécharger l'AMDEC (CSV)", am.to_csv(index=False), "amdec_road.csv")

with tab_rule:
    st.markdown("""
**Règle de maintenance conditionnelle proposée**

1. **Alarme** : le Random Forest (ou le SVM) signale une proportion de fenêtres de 10 s supérieure au seuil choisi
   → inspection du robot (charge, fixations, capteurs, puissance consommée).
2. **Surveillance** : quelques fenêtres signalées, sous le seuil d'alarme → surveiller la séquence suivante.
3. **Tendance** : le Health Index PCA reste au-dessus de 1 de façon durable → vérification du fonctionnement,
   sans conclure à une panne (indicateur peu sélectif).
4. **Toujours** : inspection périodique maintenue, car le rappel du modèle est limité (~50 %).

**Limites** : peu de séquences annotées, pas de RUL, FMDS issu de données simulées.
""")
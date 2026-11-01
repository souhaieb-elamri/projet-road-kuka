r"""RoAD KUKA LBR iiwa - modélisation finale (diagnostic + alerte précoce).

Lancer :  python run_road_final.py
Sorties : dossier final_results/ (CSV + figures)

Choix méthodologiques :
- fenêtres de 10 s (100 points à 10 Hz), 8 indicateurs temporels par capteur ;
- validation croisée GroupKFold : une séquence complète n'est jamais à la fois
  en apprentissage et en test ;
- seuil PCA choisi sur une validation interne (jamais sur le test) ;
- modèles : arbre, forêt aléatoire, SVM (supervisés) ; Isolation Forest, PCA
  (appris sur le normal uniquement) ; cascade PCA -> arbre ;
- alerte précoce : Health Index = erreur PCA / seuil, lissé (EWMA), délai de
  détection. Pas de RUL : RoAD ne contient pas de dégradation jusqu'à la panne.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from RoADDataset import Dataset

N_FEATURES = 86          # colonne 0 = ID action, 85 capteurs, colonne 86 = label
N_SENSORS = 85
WINDOW = 100             # 10 s à 10 Hz
WINDOW_SECONDS = 10
SCENARIOS = ["collision", "control", "weight", "velocity"]
STATS = ["mean", "std", "rms", "min", "max", "p2p", "energy", "kurtosis"]
QUANTILES = [0.90, 0.925, 0.95, 0.975, 0.99, 0.995]
N_FOLDS = 5
EWMA_ALPHA = 0.3
SEED = 42
OUT = Path("final_results")


# ------------------------------------------------------------------ features
def kurtosis_axis1(x):
    c = x - x.mean(axis=1, keepdims=True)
    return np.mean(c ** 4, axis=1) / (np.mean(c ** 2, axis=1) ** 2 + 1e-8)


def window_features(recording):
    sig = recording[:, :N_FEATURES].astype(np.float32)[:, 1:]
    if recording.shape[1] > N_FEATURES:
        lab = (recording[:, N_FEATURES].astype(int) > 0).astype(int)
    else:
        lab = np.zeros(len(sig), dtype=int)
    n = len(sig) // WINDOW
    if n == 0:
        return np.empty((0, N_SENSORS * len(STATS)), np.float32), np.empty(0, int)
    x = sig[: n * WINDOW].reshape(n, WINDOW, -1)
    y = lab[: n * WINDOW].reshape(n, WINDOW).max(axis=1)
    mx, mn = x.max(axis=1), x.min(axis=1)
    feats = np.concatenate([x.mean(axis=1), x.std(axis=1),
                            np.sqrt(np.mean(x ** 2, axis=1)), mn, mx, mx - mn,
                            np.mean(x ** 2, axis=1), kurtosis_axis1(x)], axis=1)
    return feats.astype(np.float32), y.astype(int)


def build_records(road):
    records = []
    for name in ["training"] + SCENARIOS:
        for rec in road.sets[name]:
            x, y = window_features(rec)
            if len(x):
                records.append({"scenario": name, "X": x, "y": y})
    return records


def sensor_names(road):
    try:
        cols = list(road.columns)
        if len(cols) >= N_FEATURES:
            return [str(c) for c in cols[1:N_FEATURES]]
    except Exception:
        pass
    return [f"sensor_{i}" for i in range(N_SENSORS)]


# ------------------------------------------------------------------- metrics
def metrics(y, pred, score=None):
    if y.sum() == 0:       # aucune anomalie dans le test : métriques non définies
        nan = float("nan")
        return {"precision": nan, "recall": nan, "f1": nan, "roc_auc": nan, "pr_auc": nan}
    d = {"precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0),
         "f1": f1_score(y, pred, zero_division=0),
         "roc_auc": float("nan"), "pr_auc": float("nan")}
    if score is not None and len(np.unique(y)) > 1:
        d["roc_auc"] = roc_auc_score(y, score)
        d["pr_auc"] = average_precision_score(y, score)
    return d


def pca_error(scaler, pca, x):
    z = scaler.transform(x)
    return np.mean((z - pca.inverse_transform(pca.transform(z))) ** 2, axis=1)


def ewma(v, alpha=EWMA_ALPHA):
    out = np.empty_like(v, dtype=float)
    acc = v[0]
    for i, val in enumerate(v):
        acc = alpha * val + (1 - alpha) * acc
        out[i] = acc
    return out


# ---------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    road = Dataset(normalize=True)
    names = sensor_names(road)
    records = build_records(road)
    print(f"{len(records)} séquences utilisables")
    for sc in ["training"] + SCENARIOS:
        r = [x for x in records if x["scenario"] == sc]
        n_w = sum(len(x["y"]) for x in r)
        n_a = sum(int(x["y"].sum()) for x in r)
        print(f"  {sc:10s}: {len(r):3d} séquences, {n_w:5d} fenêtres, {n_a:5d} anormales")

    groups = np.arange(len(records))
    n_splits = min(N_FOLDS, len(records))
    gkf = GroupKFold(n_splits=n_splits)

    fold_rows, ew_rows = [], []
    pooled = {}
    importances = np.zeros(N_SENSORS * len(STATS))
    thresholds = []
    hi_examples = {}

    def pool(model, y, pred, scen):
        d = pooled.setdefault(model, {"y": [], "pred": [], "scen": []})
        d["y"].append(y); d["pred"].append(pred); d["scen"].extend(list(scen))

    for fold, (tr, te) in enumerate(gkf.split(groups, groups=groups)):
        tr_recs = [records[i] for i in tr]
        X_tr = np.concatenate([r["X"] for r in tr_recs])
        y_tr = np.concatenate([r["y"] for r in tr_recs])
        if len(np.unique(y_tr)) < 2:
            print(f"Fold {fold}: une seule classe en apprentissage, ignoré")
            continue

        # --- validation interne pour le seuil PCA (par séquence)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        fit_i, val_i = next(gss.split(np.arange(len(tr_recs)), groups=np.arange(len(tr_recs))))
        X_fit = np.concatenate([tr_recs[i]["X"] for i in fit_i])
        y_fit = np.concatenate([tr_recs[i]["y"] for i in fit_i])
        X_val = np.concatenate([tr_recs[i]["X"] for i in val_i])
        y_val = np.concatenate([tr_recs[i]["y"] for i in val_i])

        scaler = StandardScaler().fit(X_fit[y_fit == 0])
        pca = PCA(n_components=0.95, random_state=SEED).fit(scaler.transform(X_fit[y_fit == 0]))
        err_fit = pca_error(scaler, pca, X_fit[y_fit == 0])
        err_val = pca_error(scaler, pca, X_val)
        best_q, best_f1, thr = 0.99, -1.0, None
        for q in QUANTILES:
            t = float(np.quantile(err_fit, q))
            f = f1_score(y_val, (err_val > t).astype(int), zero_division=0) if y_val.sum() else 0.0
            if f > best_f1:
                best_f1, best_q, thr = f, q, t
        thresholds.append({"fold": fold, "quantile": best_q, "threshold": thr, "val_f1": best_f1})

        # --- modèles supervisés (tous les enregistrements d'apprentissage)
        tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=3,
                                      class_weight="balanced", random_state=SEED).fit(X_tr, y_tr)
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
        svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, class_weight="balanced",
                                                  probability=False, random_state=SEED)).fit(X_tr, y_tr)
        importances += rf.feature_importances_

        # --- non supervisés : normal uniquement
        X_norm = X_tr[y_tr == 0]
        iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_norm)

        # --- test fold
        te_recs = [records[i] for i in te]
        X_te = np.concatenate([r["X"] for r in te_recs])
        y_te = np.concatenate([r["y"] for r in te_recs])
        scen_te = np.concatenate([[r["scenario"]] * len(r["y"]) for r in te_recs])

        err_te = pca_error(scaler, pca, X_te)
        pred = {
            "DecisionTree": (tree.predict(X_te), tree.predict_proba(X_te)[:, 1]),
            "RandomForest": (rf.predict(X_te), rf.predict_proba(X_te)[:, 1]),
            "SVM": (svm.predict(X_te), svm.decision_function(X_te)),
            "IsolationForest": ((iso.predict(X_te) == -1).astype(int), -iso.decision_function(X_te)),
            "PCA": ((err_te > thr).astype(int), err_te),
        }
        pred["Cascade_PCA_Tree"] = ((pred["PCA"][0] & pred["DecisionTree"][0]).astype(int), None)

        for model, (p, s) in pred.items():
            fold_rows.append({"fold": fold, "model": model, **metrics(y_te, p, s)})
            pool(model, y_te, p, scen_te)

        # --- alerte précoce par séquence de test
        for r in te_recs:
            hi = pca_error(scaler, pca, r["X"]) / thr
            sm = ewma(hi)
            raw_alarm, sm_alarm = hi > 1, sm > 1
            normal = r["y"] == 0
            row = {"fold": fold, "scenario": r["scenario"],
                   "has_anomaly": int(r["y"].any()),
                   "fa_rate_raw": float(raw_alarm[normal].mean()) if normal.any() else float("nan"),
                   "fa_rate_ewma": float(sm_alarm[normal].mean()) if normal.any() else float("nan"),
                   "delay_raw_s": float("nan"), "delay_ewma_s": float("nan"),
                   "detected_raw": 0, "detected_ewma": 0}
            if r["y"].any():
                onset = int(np.argmax(r["y"]))
                for tag, al in (("raw", raw_alarm), ("ewma", sm_alarm)):
                    after = np.where(al[onset:])[0]
                    if len(after):
                        row[f"detected_{tag}"] = 1
                        row[f"delay_{tag}_s"] = float(after[0] * WINDOW_SECONDS)
            ew_rows.append(row)
            if r["scenario"] not in hi_examples and r["y"].any():
                hi_examples[r["scenario"]] = (hi, sm, r["y"])
        print(f"Fold {fold}: seuil PCA q={best_q} ({thr:.4f}), F1 validation {best_f1:.3f}")

    # ----------------------------------------------------------- sauvegardes
    def write_csv(path, rows):
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    write_csv(OUT / "metrics_per_fold.csv", fold_rows)
    write_csv(OUT / "pca_thresholds.csv", thresholds)
    write_csv(OUT / "early_warning_per_sequence.csv", ew_rows)

    # résumé moyenne ± écart-type
    print("\n===== Résumé (moyenne ± écart-type sur les folds) =====")
    summary = []
    for model in pooled:
        rows = [r for r in fold_rows if r["model"] == model]
        entry = {"model": model}
        for k in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
            v = np.array([r[k] for r in rows], dtype=float)
            entry[k + "_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            entry[k + "_std"] = float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")
        summary.append(entry)
        print(f"{model:18s} P={entry['precision_mean']:.2f}±{entry['precision_std']:.2f} "
              f"R={entry['recall_mean']:.2f}±{entry['recall_std']:.2f} "
              f"F1={entry['f1_mean']:.2f}±{entry['f1_std']:.2f} "
              f"AUC={entry['roc_auc_mean']:.2f} PR-AUC={entry['pr_auc_mean']:.2f}")
    write_csv(OUT / "metrics_summary.csv", summary)

    # par scénario (résultats cumulés sur tous les folds)
    sc_rows = []
    print("\n===== Par scénario (cumulé) : rappel si anomalies, sinon taux de fausses alarmes =====")
    for model, d in pooled.items():
        y = np.concatenate(d["y"]); p = np.concatenate(d["pred"]); sc = np.array(d["scen"])
        for s in ["training"] + SCENARIOS:
            m = sc == s
            if not m.any():
                continue
            if y[m].sum() > 0:
                sc_rows.append({"model": model, "scenario": s, "n_windows": int(m.sum()),
                                "recall": recall_score(y[m], p[m], zero_division=0),
                                "precision": precision_score(y[m], p[m], zero_division=0),
                                "false_alarm_rate": float("nan")})
            else:
                sc_rows.append({"model": model, "scenario": s, "n_windows": int(m.sum()),
                                "recall": float("nan"), "precision": float("nan"),
                                "false_alarm_rate": float(p[m].mean())})
    write_csv(OUT / "metrics_per_scenario.csv", sc_rows)
    for r in sc_rows:
        print(f"{r['model']:18s} {r['scenario']:10s} recall={r['recall']:.2f} "
              f"prec={r['precision']:.2f} FA={r['false_alarm_rate']:.2f}")

    # alerte précoce : résumé
    print("\n===== Alerte précoce (Health Index) =====")
    for s in SCENARIOS:
        rs = [r for r in ew_rows if r["scenario"] == s]
        an = [r for r in rs if r["has_anomaly"]]
        if an:
            for tag in ("raw", "ewma"):
                det = np.mean([r[f"detected_{tag}"] for r in an])
                delays = [r[f"delay_{tag}_s"] for r in an if r[f"detected_{tag}"]]
                fa = np.nanmean([r[f"fa_rate_{tag}"] for r in rs])
                print(f"{s:10s} {tag:5s} séquences détectées={det:.2f} "
                      f"délai médian={np.median(delays) if delays else float('nan'):.0f}s "
                      f"fausses alarmes={fa:.2f}")
        elif rs:
            print(f"{s:10s} aucune anomalie ; fausses alarmes raw={np.nanmean([r['fa_rate_raw'] for r in rs]):.2f} "
                  f"ewma={np.nanmean([r['fa_rate_ewma'] for r in rs]):.2f}")

    # importances par capteur et par indicateur
    imp = importances / max(len(thresholds), 1)
    by_sensor = imp.reshape(len(STATS), N_SENSORS).sum(axis=0)
    by_stat = imp.reshape(len(STATS), N_SENSORS).sum(axis=1)
    order = np.argsort(by_sensor)[::-1][:15]
    write_csv(OUT / "importance_by_sensor.csv",
              [{"sensor": names[i], "importance": float(by_sensor[i])} for i in np.argsort(by_sensor)[::-1]])
    print("\nTop capteurs (Random Forest) :", [names[i] for i in order[:8]])
    print("Importance par indicateur :", {s: round(float(v), 3) for s, v in zip(STATS, by_stat)})

    # --------------------------------------------------------------- figures
    models = [m for m in pooled]
    fig, axs = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4))
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, models):
        cm = confusion_matrix(np.concatenate(pooled[m]["y"]), np.concatenate(pooled[m]["pred"]), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1], ["normal", "anomalie"]); ax.set_yticks([0, 1], ["normal", "anomalie"])
        ax.set_title(m, fontsize=10); ax.set_xlabel("prédit"); ax.set_ylabel("réel")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([names[i] for i in order][::-1], by_sensor[order][::-1])
    ax.set_xlabel("importance cumulée (Random Forest)"); ax.set_title("15 capteurs les plus informatifs")
    fig.tight_layout(); fig.savefig(OUT / "importance_sensors.png", dpi=150); plt.close(fig)

    if hi_examples:
        fig, axs = plt.subplots(len(hi_examples), 1, figsize=(8, 2.6 * len(hi_examples)), squeeze=False)
        for ax, (s, (hi, sm, y)) in zip(axs[:, 0], hi_examples.items()):
            t = np.arange(len(hi)) * WINDOW_SECONDS
            ax.plot(t, hi, lw=0.8, alpha=0.5, label="Health Index")
            ax.plot(t, sm, lw=1.8, label="EWMA")
            ax.axhline(1, color="r", ls="--", lw=1, label="seuil = 1")
            ax.fill_between(t, 0, hi.max() * 1.05, where=y > 0, color="orange", alpha=0.2, label="anomalie")
            ax.set_title(s); ax.set_ylabel("HI")
        axs[-1, 0].set_xlabel("temps (s)"); axs[0, 0].legend(fontsize=8, ncol=4)
        fig.tight_layout(); fig.savefig(OUT / "health_index_examples.png", dpi=150); plt.close(fig)

    print("\nRésultats dans :", OUT.resolve())


if __name__ == "__main__":
    main()

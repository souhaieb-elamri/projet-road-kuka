# Projet FMDS et Maintenance Prédictive - KUKA LBR iiwa

Ce dépôt contient l'ensemble du code source, des résultats et de l'application interactive du projet d'analyse FMDS (Fiabilité, Maintenabilité, Disponibilité, Sécurité) et de maintenance prédictive appliqué à un robot collaboratif KUKA LBR iiwa.

## 📁 Contenu du dépôt
- **Scripts Python** : 
  - `run_fmds_simule.py` : Simulation de la fiabilité (loi de Weibull, Kaplan-Meier, MLE, AIC).
  - `run_road_final.py` & `train_final_models.py` : Entraînement des modèles de Machine Learning (Random Forest, SVM) sur le dataset RoAD.
  - `app_road.py` : Application interactive développée avec Streamlit.
- **Dossiers de résultats** : 
  - `fmds_results/` : Courbes de survie, estimateurs, histogrammes.
  - `final_results/` : Matrices de confusion, métriques, exemples de Health Index.
- **Application Streamlit** : Interface de diagnostic en temps réel avec visualisation du Health Index et onglet AMDEC interactif.

## 🚀 Installation et Lancement

1. **Cloner le dépôt** :
   git clone https://github.com/TON-NOM-UTILISATEUR/projet-road-kuka.git
   cd projet-road-kuka

2. **Créer un environnement virtuel (optionnel mais recommandé)** :
   python -m venv .venv
   source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate

3. **Installer les dépendances** :
   pip install -r requirements.txt

4. **Lancer l'application interactive** :
   streamlit run app_road.py

## 📊 Dataset
Le projet utilise le jeu de données public **RoAD (Robotic Arm Dataset)**. Les données brutes ne sont pas incluses dans ce dépôt en raison de leur taille. Les séquences de démonstration sont disponibles dans le dossier `demo_sequences/`.

## ⚠️ Limites du projet
- Le dataset RoAD ne contient pas d'historique de dégradation jusqu'à la panne, il n'est donc pas possible de calculer un RUL (Remaining Useful Life) exact.
- Les données de l'analyse FMDS sont **simulées** (banc d'essai virtuel) et signalées comme telles.

---
*Projet réalisé par Souhaieb EL AMRI et Salsabil KHADHROUI - ENSIT, 3GE AII, 2026/2027.*
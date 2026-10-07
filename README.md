# ⚽ Football Expected Goals (xG) & Spatial Analytics

Pipeline complet de Machine Learning estimant la probabilité de but d'un tir en Ligue 1 à partir de données spatiales et contextuelles, avec visualisation interactive des tirs.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Viz-purple.svg)
![StatsBomb](https://img.shields.io/badge/Data-StatsBomb_API-red.svg)

---

## 📌 Fonctionnalités Clés
- **Ingestion & Extraction API** : Extraction automatisée des événements de tirs via l'API officielle `statsbombpy` sur la saison complète de Ligue 1 (+380 matchs).
- **Feature Engineering Géométrique & Spatial** :
  - Distance euclidienne réelle au centre du but.
  - Calcul trigonométrique de l'angle d'ouverture sous-tendu vers les poteaux via $\mathrm{atan2}$.
  - Prise en compte du contexte tactique (pressing défensif, phases de contre-attaque, partie du corps, type de tir).
- **Modélisation & Évaluation Probabiliste** :
  - Entraînement supervisé (Régression Logistique calibrée).
  - Évaluation par métriques de discrimination (**ROC-AUC**) et de calibration (**Log-Loss / Cross-Entropy**).
- **Dashboard Interactif Plotly** :
  - Projection dynamique des tirs sur un terrain de football réglementaire.
  - Taille des bulles indexée sur la valeur du xG et couleur selon le résultat (But / Manqué).
  - Export en application web HTML autonome (`dashboard_xg.html`).
- **Module de Scouting** :
  - Analyse du différentiel de finition ($\mathrm{Buts} - \mathrm{xG}$) pour identifier les sur-performances des attaquants.

---

## 🛠️ Stack Technique
- **Langage** : Python
- **Machine Learning & Data** : Scikit-learn, Pandas, NumPy, StatsBombPy
- **Data Visualisation** : Plotly (Express & Graph Objects)

---

## 🚀 Installation & Exécution
```bash
git clone https://github.com/kaivyyy/football-xg-ml-analytics.git
cd football-xg-ml-analytics
pip install -r requirements.txt
python app.py
from statsbombpy import sb
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# Récupérer la liste des compétitions disponibles
competitions = sb.competitions()

# Fonctions & préparation des données
# Récupérer tous les matchs de la saison 22/23 de Ligue 1
matchs = sb.matches(competition_id=7, season_id=235)
print(f"Nombre de matchs récupérés : {len(matchs)}")
print(matchs[['match_id', 'home_team', 'away_team']].head())
liste_match_ids = matchs['match_id'].tolist()

print(f"Téléchargement des tirs pour les {len(liste_match_ids)} matchs de Ligue 1...")
liste_df_tirs = []


for _, row_match in matchs.iterrows():
    m_id = row_match['match_id']
    domicile = row_match['home_team']
    exterieur = row_match['away_team']
    
    try:
        events = sb.events(match_id=m_id)
        shots = events[events['type'] == 'Shot']
        
        # On ajoute dynamiquement l'information de l'adversaire pour chaque tir
        shots['equipe_joueur'] = shots['team']
        shots['adversaire'] = shots.apply(
            lambda r: exterieur if r['team'] == domicile else domicile, axis=1
        )
        
        liste_df_tirs.append(shots)
    except Exception as e:
        continue

df_tirs_brut = pd.concat(liste_df_tirs, ignore_index=True)

def calculer_metriques_tir(x, y):
    # Calcule la distance au centre (120, 40)
    distance = math.sqrt((x - 120)**2 + (y - 40)**2)
    
    # Calcule l'angle d'ouverture réel vers les poteaux (120, 36) et (120, 44)
    numerateur = 8 * (120 - x)
    denominateur = (120 - x)**2 + (y - 36) * (y - 44)
    angle_degres = math.degrees(math.atan2(numerateur, denominateur))
    if angle_degres < 0:
        angle_degres += 180
        
    return distance, angle_degres

#Sélection des données et nettoyage : 
df_tirs = pd.DataFrame()
df_tirs['joueur'] = df_tirs_brut['player']
df_tirs['equipe'] = df_tirs_brut['equipe_joueur']
df_tirs['adversaire'] = df_tirs_brut['adversaire']
df_tirs['x'] = df_tirs_brut['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
df_tirs['y'] = df_tirs_brut['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
df_tirs['type_tir'] = df_tirs_brut['shot_type']
df_tirs['pression'] = df_tirs_brut['under_pressure'].apply(lambda x: 1 if x == True else 0)
df_tirs['contre_attaque'] = df_tirs_brut['play_pattern'].apply(lambda x: 1 if x == 'From Counter' else 0)
df_tirs['partie_corps'] = df_tirs_brut['shot_body_part']
df_tirs['but'] = df_tirs_brut['shot_outcome'].apply(lambda outcome: 1 if outcome == 'Goal' else 0)
#Si x ou y vide on supprime
df_tirs = df_tirs.dropna(subset=['x', 'y'])
# 1. Filtre métier : On ne garde que les tirs pris dans les 45 derniers mètres (X > 75)
df_tirs_propres = df_tirs[df_tirs['x'] > 75]

df_tirs_propres[['distance', 'angle']] = df_tirs_propres.apply(lambda row: calculer_metriques_tir(row['x'], row['y']), axis=1, result_type='expand')
print(f"Nombre de tirs après filtrage : {len(df_tirs_propres)}")

df_features = pd.get_dummies(df_tirs_propres[['distance', 'angle', 'type_tir', 'pression', 'contre_attaque', 'partie_corps']], columns=['type_tir', 'partie_corps'])
print("\nNouvelles features après encodage :")
print(df_features.columns.tolist())
# 2. Séparation des buts et des non-buts
buts = df_tirs_propres[df_tirs_propres['but'] == 1]
non_buts = df_tirs_propres[df_tirs_propres['but'] == 0]

print(df_tirs_propres[['joueur', 'distance', 'angle', 'but']].head())


X = df_features
y = df_tirs_propres['but']

# 2. Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entraînement de la nouvelle IA
modele_xg_evolution = LogisticRegression()
modele_xg_evolution.fit(X_train, y_train)

# 4. Évaluation
probabilites_test = modele_xg_evolution.predict_proba(X_test)[:, 1]
print("\n--- NOUVELLES PERFORMANCES IA (AVEC CONTEXTE) ---")
print(f"Nouveau Log Loss : {log_loss(y_test, probabilites_test):.4f}")
print(f"Nouveau Score AUC-ROC : {roc_auc_score(y_test, probabilites_test):.4f}")

# 5. Nouvelle fonction de prédiction contextuelle
def predire_xg_contexte(x, y, type_tir, pression, contre_attaque, partie_corps):
    distance, angle = calculer_metriques_tir(x, y)
    
    # On crée un dictionnaire avec toutes nos features initialisées à 0
    donnees_tir = {col: 0.0 for col in X.columns}
    
    # On remplit les valeurs géométriques
    donnees_tir['distance'] = distance
    donnees_tir['angle'] = angle
    donnees_tir['pression'] = float(pression)
    donnees_tir['contre_attaque'] = float(contre_attaque)
    
    # On active l'interrupteur du type de tir s'il existe dans nos colonnes
    nom_colonne_type = f"type_tir_{type_tir}"
    if nom_colonne_type in donnees_tir:
        donnees_tir[nom_colonne_type] = 1.0

    nom_colonne_partie_corps = f"partie_corps_{partie_corps}"
    if nom_colonne_partie_corps in donnees_tir:
        donnees_tir[nom_colonne_partie_corps] = 1.0
    # Conversion en DataFrame pour Scikit-Learn
    df_un_tir = pd.DataFrame([donnees_tir])
    
    # Prédiction
    return modele_xg_evolution.predict_proba(df_un_tir)[:, 1][0]

# --- RE-TEST DU PENALTY ---
xg_penalty_neuf = predire_xg_contexte(108, 40, 'Penalty', '0', '0', 'Foot')
xg_open_play_neuf = predire_xg_contexte(108, 40, 'Open Play', '1', '1', 'Head')

print("\n--- COMPARAISON À POSITION ÉGALE (12m face au but) ---")
print(f"xG pour un Penalty sans Pression Défensive: {xg_penalty_neuf:.4f}")
print(f"xG pour une action de jeu classique (Open Play, avec Pression Défensive) : {xg_open_play_neuf:.4f}")


import plotly.graph_objects as go
import plotly.express as px

# 1. On s'assure que les xG sont calculés sur notre dataset propre
df_tirs_propres['mon_xG'] = df_tirs_propres.apply(
    lambda row: predire_xg_contexte(row['x'], row['y'], row['type_tir'], row['pression'], row['contre_attaque'], row['partie_corps']), 
    axis=1
)

# 2. On sépare pour gérer les couleurs (Rouge pour But, Bleu pour Manqué)
df_tirs_propres['Resultat'] = df_tirs_propres['but'].apply(lambda x: 'BUT' if x == 1 else 'Tir manqué / arrêté')

# 3. Création du scatter plot interactif avec Plotly
fig = px.scatter(
    df_tirs_propres,
    x='x',
    y='y',
    size='mon_xG',          # La taille de la bulle dépend de ton xG
    color='Resultat',        # Couleur selon le résultat
    color_discrete_map={'BUT': '#ff4d6d', 'Tir manqué / arrêté': '#00b4d8'},
    hover_name='joueur',     # Le titre de l'infobulle (Nom du joueur)
    # Les informations affichées au survol :
    hover_data={
        'equipe': True,
        'adversaire': True,
        'x': False,          # On cache les coordonnées brutes pour ne pas polluer
        'y': False,
        'mon_xG': ':.3f',    # On affiche l'xG arrondi à 3 décimales
        'distance': ':.1f',  # Distance en unités
        'angle': ':.1f',     # Angle en degrés
        'pression': True,    # 0 ou 1
        'contre_attaque': True,    # 0 ou 1
        'type_tir': True     # Open Play, Penalty...
    },
    title="Dashboard Interactif xG - Ligue 1 22/23"
)

# 4. TRACÉ DU TERRAIN EN ARRIÈRE-PLAN (Format StatsBomb 120x80)
# On restreint la vue aux 45 derniers mètres (X de 75 à 120, Y de 0 à 80)
fig.update_layout(
    xaxis=dict(range=[75, 125], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[-5, 85], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
    template="plotly_dark",
    paper_bgcolor='#1e1e1e',
    plot_bgcolor='#1e1e1e',
    width=1000,
    height=700
)

# Ajout des lignes du terrain (Ligne de fond, surface de réparation, cages)
shapes = [
    # Ligne de fond droite (X=120)
    dict(type="line", x0=120, y0=0, x1=120, y1=80, line=dict(color="white", width=2)),
    # Grande surface de réparation (X de 102 à 120, Y de 18 à 62)
    dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="white", width=2)),
    # Petite surface / Six mètres (X de 114 à 120, Y de 30 à 50)
    dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="white", width=2)),
    # Les cages (X=120, Y de 36 à 44)
    dict(type="rect", x0=120, y0=36, x1=121, y1=44, line=dict(color="white", width=3), fillcolor="white"),
    # Point de penalty (X=108, Y=40)
    dict(type="circle", x0=107.8, y0=39.8, x1=108.2, y1=40.2, fillcolor="white", line=dict(color="white"))
]
fig.update_layout(shapes=shapes)

# 5. SAUVEGARDE ET AFFICHAGE
# Génère un fichier HTML interactif autonome sur ton bureau
fig.write_html("dashboard_xg.html")
print("\nLe dashboard interactif a été sauvegardé sous le nom 'dashboard_xg.html'. Ouvre-le avec ton navigateur !")
fig.show()


#=============SCOOTING===================


# 1. On groupe par joueur et on fait la somme des buts et de TES xG
scouting_df = df_tirs_propres.groupby(['joueur', 'equipe']).agg(
    buts_reels=('but', 'sum'),
    xg_totaux=('mon_xG', 'sum'),
    total_tirs=('but', 'count')
).reset_index()

# 2. On calcule la différence de finition
scouting_df['performance_finition'] = scouting_df['buts_reels'] - scouting_df['xg_totaux']

# 3. On ne garde que les joueurs qui ont tenté au moins 10 tirs pour avoir un échantillon sérieux
buteurs_reguliers = scouting_df[scouting_df['total_tirs'] >= 10]

# 4. On trie du meilleur finisher au moins bon
top_finishers = buteurs_reguliers.sort_values(by='performance_finition', ascending=False)

print("\n--- TOP 5 DES MEILLEURS FINISHERS DE NOTRE DATASET ---")
print(top_finishers[['joueur', 'equipe', 'buts_reels', 'xg_totaux', 'performance_finition']].head(5))

print("\n--- TOP 5 DES JOUEURS EN SOUS-PERFORMANCE (SOUS-FINITION) ---")
print(top_finishers[['joueur', 'equipe', 'buts_reels', 'xg_totaux', 'performance_finition']].tail(5))
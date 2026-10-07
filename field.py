from math import sqrt, degrees, atan2
import pandas as pd

donnees_tirs = {
    'joueur': ['K. Mbappé', 'B. Bourigeaud', 'O. Dembélé'],
    'x': [108.0, 112.0, 95.0],
    'y': [40.0, 55.0, 25.0]
}

df = pd.DataFrame(donnees_tirs)

def calculer_metriques_tir(x,y):
    distance = sqrt((x-120)**2 + (y-40)**2)
    numerateur = 8 * (120 - x)
    denominateur = (120 - x)**2 + (y - 36) * (y - 44)

    angle_radians = atan2(numerateur, denominateur)
    angle_degres = degrees(angle_radians)

    # Cas extrême si le joueur tire de derrière la ligne (cas extrême), 
    # l'angle peut être négatif. 
    # On s'assure qu'il reste positif :
    if angle_degres < 0:
        angle_degres += 180
    return (distance, angle_degres)

df[['distance', 'angle']] = df.apply(lambda row: calculer_metriques_tir(row['x'], row['y']), axis=1, result_type='expand')

print(df)

df['but'] = [1,0,0]

X = df[['distance', 'angle']]
y = df['but']
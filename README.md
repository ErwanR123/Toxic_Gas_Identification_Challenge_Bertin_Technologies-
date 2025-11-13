# 🚀 Toxic Gas Detection – Domain Shift Regression Pipeline  
### Challenge Data ENS x Bertin Technologies 2025

Ce dépôt présente un pipeline complet de **régression multi-sorties** pour la détection de **23 gaz** à partir de données capteurs, dans un contexte marqué par un **domain shift dû à l’humidité**.  
L’objectif est de construire un modèle robuste capable de généraliser entre des distributions `train` et `test` hétérogènes.

---

## 📌 1. Problématique

Le dispositif ChemProX enregistre plusieurs signaux :

- **8 canaux IMS** (M4–M7, M12–M15), très sensibles à l’humidité,  
- **Capteurs auxiliaires** (S1–S3), plus stables,  
- **Variable environnementale** : `Humidity`.

L’analyse exploratoire montre un **désalignement important entre train et test**, causé par :

- une forte dérive des signaux en fonction de `Humidity`,  
- des régimes d’humidité très différents entre les deux jeux,  
- un vrai **covariate/domain shift** dégradant les modèles classiques.

---

## 💧 2. Dé-humidification (External Dehumidification)

Avant le feature engineering, chaque canal IMS est corrigé via une :

### ✔ Régression **Ridge polynomiale (degré 2)** capteur-par-capteur

Pour chaque capteur $( X_j )$ :

$$ X_j^{\text{resid}} = X_j - \hat{f}_j(H)$$

Caractéristiques :

- correction **indépendante** pour chaque capteur IMS,  
- version **transductive** : ajustement sur la distribution jointe `train + test`,  
- les signaux **bruts** et **dé-humidifiés** sont tous conservés dans le modèle,  
- `Humidity` et `H²` sont retirés avant l’apprentissage pour éviter les shortcuts.

Cette étape réduit significativement le shift train→test.

---

## 📐 3. Feature Engineering (géométrie du réseau de capteurs)

Le FE exploite la **structure spatiale** des capteurs IMS (4 par bloc).

### Principales briques :

- **Vues normalisées** : L2 intra-échantillon  
- **Descripteurs locaux** : gradients (Δ), courbure centrale (Δ²), log-ratios adjacents  
- **Contrastes inter-blocs**  
- **Moments de forme** : centroid, bandwidth  
- **Statistiques globales** : mean, std, énergie, IQR, amplitude de pic  
- **Similarité directionnelle** entre blocs : cosinus (sur vue L2)

Ces descripteurs capturent la forme et la cohérence interne du signal plutôt que sa seule amplitude.

---

## 🌲 4. Modèle final : Random Forest robuste au domain shift

Modèle utilisé : **RandomForestRegressor multi-sorties**  
Hyperparamètres principaux :
n_estimators = 2200
max_depth = 16
min_samples_leaf = 65
max_features = 0.36
bootstrap = True
max_samples = 0.74

Pourquoi ce choix ?

- robuste aux non-linéarités et bruit capteur,  
- insensible à l’échelle des features,  
- très stable sous dérive d’environnement.

Validation : **stratification par régimes d’humidité**.

---

## 🧪 5. Pistes exploratoires testées

Plusieurs alternatives ont été évaluées :

- modèles locaux par zones d’humidité (experts low/mid/high),  
- **importance weighting** pour compenser le déséquilibre train/test,  
- vues robustes (MAD),  
- prototypes KMeans sur les formes,  
- boosting (CatBoost / XGBoost) → instables sous domain shift,  
- correction interne H-residual → moins efficace que la dé-humidification externe.

Aucune n’a surpassé la combinaison :

> **Dé-humidification externe + FE géométrique + Random Forest**

---

## 🎯 7. Résultats

- Réduction majeure du domain shift  
- Modèle final stable même en humidité extrême  
- Score public leaderboard : **0.14318**
- Classement actuel sur la plateforme du challenge: **10ᵉ sur 148**
  → [Voir le classement public](https://challengedata.ens.fr/participants/challenges/156/ranking/public)  
- Pipeline modulaire réutilisable sur tout jeu de données capteurs

---
Contact

Projet réalisé par Erwan Ouabdesselam
Master MS2A — Sorbonne Université

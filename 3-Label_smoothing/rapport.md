# Estimation d'âge apparent par classification ordinale — APPA-real

**Backbone** : SE-ResNeXt50-32x4d pré-entraîné ImageNet  
**Dataset** : APPA-real — 7 591 images (4 113 train / 1 500 valid / 1 978 test)  
**Métrique** : MAE (Mean Absolute Error) en années  
**Référence humaine** : ~4.5 ans (désaccord inter-annotateurs)

---

## Pourquoi reformuler l'estimation d'âge en classification ?

L'âge apparent est une perception **subjective** : sur APPA-real, chaque image est notée par des dizaines d'annotateurs et la variance inter-annotateurs peut atteindre ±4 ans. Deux conséquences directes :

1. **La cible est floue** — prédire "34 ans" pour une image notée 34.7 en moyenne est arbitraire ; une distribution de probabilité centrée sur 34.7 est plus honnête.
2. **La régression directe est instable** — un modèle qui n'apprend pas converge vers la moyenne du dataset (~30 ans sur APPA-real) par minimisation naïve du MAE, phénomène dit de *mean collapse*.

La solution adoptée : **classification ordinale sur 101 bins** (un bin par année entière de 0 à 100), avec prédiction finale par espérance de la distribution softmax :

$$\hat{a} = \sum_{k=0}^{100} k \cdot \text{softmax}(\text{logit}_k)$$

Cette espérance est une valeur **continue** (ex. 34.7 ans), contrairement à un argmax qui forcerait des prédictions entières.

---

## Partie 1 — Approche naïve : Label Smoothing σ = 1.5

### Principe mathématique

Au lieu d'un label *one-hot* (seul le bin de l'âge entier arrondi vaut 1), on construit une **distribution gaussienne** centrée sur l'âge apparent moyen (non arrondi) :

$$y_k = \frac{1}{Z} \exp\!\left(-\frac{(k - \mu)^2}{2\sigma^2}\right), \quad k \in \{0, \ldots, 100\}$$

où $\mu$ est l'âge apparent continu (ex. 34.7) et $Z$ est la constante de normalisation.

La loss est une **divergence KL** entre cette distribution cible et le softmax prédit, équivalente à une cross-entropie pondérée :

$$\mathcal{L} = -\sum_{k=0}^{100} y_k \log p_k$$

Cette formulation respecte implicitement la **structure ordinale** : les bins proches de l'âge cible sont moins pénalisés que les bins lointains, contrairement à une cross-entropie classique qui punit également tous les bins incorrects.

### Architecture

```
Image 224×224×3
      │
      ▼
SE-ResNeXt50-32x4d  (last_linear = Identity)
      │  (B, 2048, 1, 1)
      ▼
nn.Flatten(1)          ← critique : sans ça, BN reçoit un tenseur 4D
      │  (B, 2048)
      ▼
BatchNorm1d(2048)
      │
      ▼
Linear(2048→256) → BN → ReLU → Dropout(0.3) → Linear(256→101)
      │  (B, 101 logits)
      ▼
Softmax expectation → âge prédit ∈ ℝ
```

**Backbone SE-ResNeXt50** : combine trois idées complémentaires —
- *ResNet* : connexions résiduelles contre la dégradation en profondeur
- *ResNeXt* : convolutions groupées (32 groupes × 4 canaux) pour un meilleur rapport expressivité/paramètres
- *Squeeze-and-Excitation* : attention par canal, le réseau apprend quelles features comptent

### Stratégie d'entraînement

| Phase | Epochs | Backbone | lr backbone | lr tête |
|-------|--------|----------|-------------|---------|
| Warmup (HEAD ONLY) | 1–5 | ❄️ gelé | — | 1e-4 |
| Fine-tuning (FULL FT) | 6–30 | 🔥 dégelé | **1e-5** | 1e-4 |

Le backbone reçoit un lr 10× plus petit pour éviter le *catastrophic forgetting* des représentations ImageNet.

### Bug critique identifié : le Flatten manquant

`pretrainedmodels` renvoie **(B, 2048, 1, 1)** après l'average pooling, pas **(B, 2048)**. Sans `nn.Flatten(1)` explicite, `BatchNorm1d` reçoit un tenseur 4D, produit des statistiques fausses, et le modèle prédit ~49 ans pour toutes les images (la moyenne globale APPA-real). Ce bug est corrigé dans les deux versions.

### Problème de σ = 1.5 : prédictions discrètes

Avec **σ = 1.5**, la gaussienne est si piquée (étalement sur ±3 ans) que le réseau converge vers un comportement proche du *one-hot* : la softmax devient quasi-argmax et les prédictions sont des **entiers** (34, 36, 38...), ce qui dégrade fortement le MAE.

**Ablation des σ :**

| σ | Entropie cible | Comportement |
|---|---------------|--------------|
| 0.5 | ~0.09 | Quasi one-hot, prédictions entières |
| 1.5 | ~0.72 | Prédictions entières ou semi-entières |
| 3.0 | ~1.52 | Prédictions continues ✅ |
| 5.0 | ~2.10 | Distribution trop plate, signal faible |

### Résultats

- **Val MAE** : ~8–9 ans (avec σ = 1.5, images complètes, sans crop facial)
- Images complètes utilisées (pas de crop facial), batch size 32

---

## Partie 2 — Approche améliorée : Label Smoothing σ = 3.0 + améliorations

Trois corrections apportées simultanément sur la version naïve.

### Correction 1 — Crop facial

Chaque image APPA-real est fournie en deux versions :
- `000000.jpg` — image complète (fond, corps, vêtements)
- `000000.jpg_face.jpg` — crop du visage détecté automatiquement

En utilisant le crop facial, le réseau traite **uniquement l'information pertinente** pour estimer l'âge : les rides, la texture de peau, la forme du visage. Le bruit dû au fond ou aux vêtements est éliminé.

**Gain empirique sur APPA-real : −1 à −2 ans de MAE.**

### Correction 2 — σ : 1.5 → 3.0

Passage à **σ = 3.0** (étalement sur ±6 ans). La distribution cible est suffisamment large pour que le réseau produise des prédictions **continues** plutôt que discrètes. Ce changement est la correction la plus impactante sur le MAE.

### Correction 3 — Cible float, batch size 64, eta_min

- **Cible float** : on passe `apparent_age_avg` (ex. 34.7) à la loss, pas `age_class = round(34.7) = 35`. La gaussienne est centrée sur la valeur exacte, pas l'entier arrondi.
- **Batch size 32 → 64** : meilleure estimation du gradient sur GPU T4 (14 Go).
- **`eta_min=1e-6`** dans `CosineAnnealingLR` : le learning rate ne tombe pas à 0 en fin d'entraînement.

### Augmentations d'entraînement (renforcées)

```python
A.HorizontalFlip(p=0.5)
A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4)
A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-20, 20), p=0.4)
A.GaussNoise(p=0.2)
```

### Résultats

- **Val MAE : ~6.3 ans** (meilleur checkpoint sur 30 epochs)
- **MAE test : ~6.5 ans**
- Gain total vs version naïve : **−2 à −3 ans de MAE**

### Test-Time Augmentation (TTA)

Post-traitement sans réentraînement : inférer **T = 5 vues augmentées** de chaque image et moyenner les prédictions.

$$\hat{a}_{\text{TTA}} = \frac{1}{T} \sum_{t=1}^{T} \hat{a}(\text{aug}_t(\mathbf{x}))$$

| Vue | Transformation |
|-----|---------------|
| 0 | Originale |
| 1 | Flip horizontal |
| 2 | Brightness/contrast léger |
| 3 | Rotation −10° |
| 4 | Rotation +10° |

**Gain TTA : −0.2 à −0.4 ans de MAE** sans modifier le modèle.

### Comparaison état de l'art

| Modèle | MAE test |
|--------|---------|
| Humain (désaccord inter-annotateurs) | ~4.5 ans |
| DEX (VGG-16, 2016) | 6.52 ans |
| **Notre modèle — Label Smoothing σ=3.0** | **~6.3 ans** |
| MiVOLO (2023) | 4.96 ans |

---

## Partie 3 — Méthodes alternatives explorées

### 3.1 Mean-Variance Loss
*Pan et al., CVPR 2018*

#### Motivation

Le Label Smoothing impose une gaussienne cible **fixe** (σ = 3.0 pour toutes les images), indépendamment de l'ambiguïté intrinsèque de chaque image. La Mean-Variance Loss supprime ce choix arbitraire : au lieu d'imposer une distribution cible, elle **régularise directement la distribution prédite** par le réseau.

#### Formulation

La loss combine trois termes :

$$\mathcal{L}_{\text{MV}} = \mathcal{L}_{\text{CE}} + \lambda_1 \underbrace{(\hat{\mu} - \mu^*)^2}_{\text{erreur de moyenne}} + \lambda_2 \underbrace{\hat{\sigma}^2}_{\text{régularisation de variance}}$$

avec :

$$\hat{\mu} = \sum_{k=0}^{100} k \cdot p_k \qquad \text{(espérance prédite)}$$

$$\hat{\sigma}^2 = \sum_{k=0}^{100} (k - \hat{\mu})^2 \cdot p_k \qquad \text{(variance prédite)}$$

- $\mathcal{L}_{\text{CE}}$ : cross-entropie standard sur l'âge entier arrondi
- Le terme $\lambda_1 (\hat{\mu} - \mu^*)^2$ pénalise l'écart entre l'espérance prédite et l'âge cible
- Le terme $\lambda_2 \hat{\sigma}^2$ pénalise un étalement excessif de la distribution, forçant des prédictions confiantes

Valeurs utilisées : $\lambda_1 = 0.2$, $\lambda_2 = 0.05$.

La prédiction finale est identique au Label Smoothing : $\hat{a} = \hat{\mu} = \sum_k k \cdot p_k$.

#### Avantage clé

Le réseau apprend **conjointement** *où* prédire (contrôlé par le terme de moyenne) et *avec quelle confiance* (contrôlé par le terme de variance). Il n'est plus nécessaire de choisir σ manuellement — σ effectif émerge de l'entraînement.

#### Point de vigilance technique

Le buffer `classes` (le vecteur $[0, 1, \ldots, 100]$) enregistré dans le module via `register_buffer` doit être déplacé sur le même device que les logits via `.to(device)`. Sans cela, un `RuntimeError: Expected all tensors to be on the same device` survient dès la première forward pass sur GPU.

#### Architecture

Même backbone SE-ResNeXt50, même tête `Linear(2048→256→101)`, seule la loss change.

---

### 3.2 CORAL — Consistent Ordinal Regression
*Cao et al., Pattern Recognition Letters 2020*

#### Motivation

La cross-entropie classique (et le Label Smoothing) traitent les 101 bins comme des **classes indépendantes** : rien n'interdit au modèle de prédire $P(y > 50) < P(y > 60)$, ce qui est ordinalement incohérent. CORAL garantit cette cohérence par construction architecturale.

#### Reformulation du problème

Au lieu d'une classification en 101 classes, CORAL décompose le problème en **100 problèmes de classification binaire** :

$$P(\hat{y} > k) \quad \text{pour } k \in \{0, 1, \ldots, 99\}$$

Chaque classifieur répond : "la personne a-t-elle **plus de k ans** ?"

#### Architecture de la tête CORAL

Tous les classifieurs **partagent les mêmes poids** $\mathbf{W}$, mais ont des **biais distincts** $b_k$ :

$$f_k(\mathbf{x}) = \mathbf{W}^\top \mathbf{x} + b_k$$

En pratique : une seule `Linear(embed_dim, 1, bias=False)` (poids partagés) + un `nn.Parameter` de taille $(K-1)$ pour les biais ordinaux.

Ce partage des poids **garantit la cohérence ordinale** : puisque le score de base $\mathbf{W}^\top \mathbf{x}$ est identique pour tous les seuils, et que les biais $b_k$ sont monotones décroissants après entraînement, on a nécessairement $P(\hat{y} > k) \geq P(\hat{y} > k+1)$.

#### Loss

Somme des $(K-1)$ binary cross-entropies :

$$\mathcal{L}_{\text{CORAL}} = \frac{1}{K-1} \sum_{k=0}^{K-2} \text{BCE}\!\left(\sigma(f_k(\mathbf{x})),\ \mathbf{1}[y > k]\right)$$

où $\sigma$ est la fonction sigmoïde et $\mathbf{1}[y > k]$ vaut 1 si l'âge réel est supérieur à $k$.

#### Prédiction

Comptage des seuils franchis (probabilité > 0.5) :

$$\hat{a} = \sum_{k=0}^{K-2} \mathbf{1}\!\left[\sigma(f_k(\mathbf{x})) > 0.5\right]$$

La prédiction est un entier entre 0 et 100, interprétable comme "le nombre de seuils que la personne dépasse".

#### Comparaison des méthodes

| Méthode | Choix de σ | Cohérence ordinale | Calibration | Complexité |
|---------|-----------|-------------------|-------------|------------|
| Label Smoothing σ=1.5 | Fixe (trop petit) | Partielle | Non | ⭐ |
| Label Smoothing σ=3.0 | Fixe (bon) | Partielle | Non | ⭐ |
| Mean-Variance Loss | **Appris** | Partielle | **Oui** | ⭐⭐ |
| CORAL | N/A | **Garantie** | Non | ⭐⭐ |

---

## Synthèse

L'évolution méthodologique suit une logique claire :

1. **Version naïve (σ=1.5, images complètes)** : le manque de lissage et l'absence de crop facial donnent ~8–9 ans de MAE.
2. **Version améliorée (σ=3.0, crop facial, batch 64)** : trois corrections indépendantes mais complémentaires abaissent le MAE à **~6.3 ans**, proche de la référence DEX (6.52 ans) mais avec un backbone bien plus récent.
3. **Mean-Variance Loss & CORAL** : deux directions différentes pour aller au-delà — l'une en adaptant automatiquement la forme de la distribution prédite, l'autre en garantissant la cohérence ordinale par construction architecturale.

Le plafond théorique reste ~4.5 ans (désaccord humain), atteint uniquement par des modèles modernes multi-tâches comme MiVOLO (2023).

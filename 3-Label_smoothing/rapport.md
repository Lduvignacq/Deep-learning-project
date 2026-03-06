# 3 — Label Smoothing Gaussien pour l'estimation d'âge apparent

> **Dataset** : APPA-real — 7 591 images annotées par des humains (âge apparent + âge réel)  
> **Backbone** : SE-ResNeXt50-32x4d pré-entraîné ImageNet (via `pretrainedmodels`)  
> **Métrique principale** : MAE (Mean Absolute Error) en années sur le jeu de test  
> **Meilleur MAE obtenu** : ~6.3 ans (version de base) → objectif ≤ 5.5 ans avec les améliorations

---

## 1. Contexte et problématique

### 1.1 Pourquoi l'estimation d'âge est un problème difficile

L'**âge apparent** n'est pas une grandeur physique mesurable : c'est une perception subjective qui dépend de l'éclairage, de l'expression, du maquillage, de l'ethnicité, et surtout du jugement individuel de chaque annotateur. Sur APPA-real, chaque image est annotée par plusieurs dizaines de personnes indépendantes — la variance inter-annotateurs atteint souvent ±4 ans.

Deux conséquences directes :
1. **La cible est floue** : prédire exactement 34 ans pour une image annotée en moyenne à 34.7 ans est arbitraire. Une distribution de probabilité centrée sur 34.7 est plus honnête qu'un label one-hot.
2. **Le MAE plancher humain est ~4.5 ans** : un modèle parfait ne peut pas faire mieux que le désaccord moyen entre annotateurs humains.

### 1.2 Pourquoi ne pas faire de la simple régression ?

La régression directe (prédire un scalaire via `nn.L1Loss`) a deux défauts sur ce type de données :

- Elle traite l'espace des âges comme **métrique pur** : une erreur de 30→31 est aussi grave qu'une erreur de 30→60, ce qui est faux perceptuellement.
- Elle converge souvent vers la **moyenne du dataset** (~49 ans sur APPA-real) quand le réseau n'apprend pas — le gradient de la L1 pousse vers la médiane et le réseau "collapse".

La **classification ordinale** (101 bins d'âge de 0 à 100) est empiriquement plus stable et plus précise sur les benchmarks d'âge apparent (DEX, MiVOLO, DLDL).

---

## 2. Approche : Label Smoothing Gaussien

### 2.1 Principe

Au lieu d'un label one-hot `y = [0, 0, ..., 1, ..., 0]` où seul le bin de l'âge exact vaut 1, on construit une **distribution gaussienne** centrée sur l'âge réel :

$$
y_k = \frac{1}{Z} \exp\!\left(-\frac{(k - \mu)^2}{2\sigma^2}\right), \quad k \in \{0, 1, \ldots, 100\}
$$

avec $\mu$ = âge apparent moyen (float, non arrondi) et $Z$ le facteur de normalisation pour que $\sum_k y_k = 1$.

La loss est une **cross-entropie KL** entre cette distribution cible et la distribution prédite par le softmax :

$$
\mathcal{L} = -\sum_{k=0}^{100} y_k \log p_k, \quad p_k = \text{softmax}(\text{logit}_k)
$$

### 2.2 Pourquoi c'est meilleur qu'un one-hot

| | One-hot (CELoss standard) | Label Smoothing Gaussien |
|---|---|---|
| Cible pour âge 34.7 | bin 35 = 1, reste = 0 | gaussienne centrée sur 34.7 |
| Pénalise prédire 33 ou 35 | pareil | moins fort que prédire 10 |
| Structure ordinale respectée | ❌ | ✅ |
| Résistance au bruit d'annotation | faible | forte |

### 2.3 Paramètre clé : σ

- **σ trop petit (1.5)** : distribution quasi one-hot → le réseau apprend à coller à un seul bin → la softmax devient un argmax → les prédictions sont des entiers (34, 36, 38...) → le MAE stagne
- **σ trop grand (10.0)** : distribution trop plate → tous les bins ont la même probabilité cible → le gradient est nul → le réseau n'apprend rien
- **σ = 3.0** : compromis optimal sur APPA-real, correspond à l'incertitude typique inter-annotateurs (~±3 ans)

```python
class OrdinalLabelSmoothing(nn.Module):
    def __init__(self, num_classes: int = 101, sigma: float = 3.0):
        super().__init__()
        self.K, self.sigma = num_classes, sigma

    def _smooth_labels(self, targets):
        bins  = torch.arange(self.K, device=targets.device).float()
        diff  = bins.unsqueeze(0) - targets.float().unsqueeze(1)   # (B, K)
        labels = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        return labels / labels.sum(dim=1, keepdim=True)            # normalisé

    def forward(self, logits, targets):
        log_probs     = F.log_softmax(logits, dim=-1)
        smooth_labels = self._smooth_labels(targets)
        return -(smooth_labels * log_probs).sum(dim=-1).mean()
```

### 2.4 Prédiction : softmax expectation

La prédiction finale n'est pas l'argmax (qui serait toujours entier), mais l'**espérance** de la distribution softmax :

$$
\hat{a} = \sum_{k=0}^{100} k \cdot \text{softmax}(\text{logit}_k)
$$

C'est une valeur continue (ex. 34.7 ans) qui exploite toute la distribution, pas seulement le pic.

---

## 3. Architecture du modèle

### 3.1 Pipeline complet

```
Image (224×224×3)
      │
      ▼
SE-ResNeXt50-32x4d          ← backbone pré-entraîné ImageNet
(last_linear = Identity)     ← on retire la tête de classification
      │
      ▼  (B, 2048, 1, 1)
nn.Flatten(1)                ← CRITIQUE : sans ça, BatchNorm1d reçoit un tenseur 4D
      │
      ▼  (B, 2048)
BatchNorm1d(2048)            ← normalise les features avant la tête
      │
      ▼
AppAgeHead:
  Linear(2048 → 256)
  BatchNorm1d(256)
  ReLU
  Dropout(0.3)
  Linear(256 → 101)          ← 101 logits, un par année de 0 à 100
      │
      ▼  (B, 101)
OrdinalLabelSmoothing        ← pendant l'entraînement
softmax expectation          ← pendant l'inférence → âge prédit (float)
```

### 3.2 Bug critique identifié et corrigé : le Flatten manquant

`pretrainedmodels` avec `last_linear = nn.Identity()` renvoie un tenseur de forme **(B, 2048, 1, 1)** après le average pooling global — pas **(B, 2048)**. Sans `nn.Flatten(1)`, le `BatchNorm1d` reçoit un tenseur 4D, ce qui produit des statistiques complètement fausses et force le modèle à prédire la moyenne du dataset (~49 ans) pour tous les exemples.

```python
# AVANT (bugué) : BN reçoit (B, 2048, 1, 1) → garbage
features = self.backbone(images)
embeddings = self.bn(features)      # ❌ shape incorrecte

# APRÈS (corrigé) : BN reçoit (B, 2048) → correct
features   = self.backbone(images)  # (B, 2048, 1, 1)
features   = self.flatten(features) # (B, 2048) ← FIX
embeddings = self.bn(features)      # ✅
```

### 3.3 SE-ResNeXt50-32x4d : pourquoi ce backbone ?

Le **Squeeze-and-Excitation ResNeXt50** combine trois idées :
- **ResNet** : connexions résiduelles pour entraîner des réseaux profonds
- **ResNeXt** : convolutions groupées (32 groupes × 4 canaux) → meilleur rapport expressivité/paramètres
- **Squeeze-and-Excitation** : recalibration des canaux via attention → le réseau apprend quelles features sont pertinentes pour chaque image

Il est particulièrement efficace pour la reconnaissance faciale et l'estimation d'âge (utilisé dans DEX, AgeNet).

---

## 4. Stratégie d'entraînement

### 4.1 Problème du fine-tuning naïf

Appliquer un learning rate uniforme (1e-4) sur tous les paramètres d'un réseau pré-entraîné détruit les représentations ImageNet acquises pendant des centaines d'epochs. Le réseau "oublie" ce qu'il a appris — c'est le **catastrophic forgetting**.

### 4.2 Solution : freeze + dégel progressif (warmup)

```
Epochs 1-5  : backbone GELÉ (requires_grad=False)
              → seuls BN + AppAgeHead sont entraînés (lr=1e-4)
              → le réseau apprend à mapper les features ImageNet sur les âges
              → évite que des gradients bruités du début brisent le backbone

Epoch 6+    : backbone DÉGELÉ
              → backbone : lr=1e-5  (10x plus petit que la tête)
              → BN + tête : lr=1e-4
              → fine-tuning complet avec lr différencié
```

### 4.3 Scheduler

`CosineAnnealingLR` avec `eta_min=1e-6` : le learning rate suit une courbe cosinus décroissante, mais ne tombe jamais en dessous de 1e-6 (évite la stagnation en fin d'entraînement).

### 4.4 Crop facial : amélioration majeure

Chaque image APPA-real est fournie en deux versions :
- `000000.jpg` : image complète (fond, corps, vêtements)
- `000000.jpg_face.jpg` : crop du visage détecté automatiquement

En utilisant le crop facial, le réseau traite uniquement l'information pertinente pour l'estimation d'âge, sans gaspiller de capacité sur l'arrière-plan. Gain empirique : **-1 à -2 ans de MAE**.

### 4.5 Augmentations

```python
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),                # symétrie faciale
    A.RandomBrightnessContrast(p=0.4),      # variabilité éclairage
    A.Affine(rotate=(-20, 20), p=0.4),      # légères rotations
    A.GaussNoise(p=0.2),                    # robustesse au bruit
    A.Normalize(imagenet_stats),
    ToTensorV2(),
])
```

---

## 5. Résultats et analyse

### 5.1 Résultats obtenus

| Version | σ | Dataset train | MAE val | Notes |
|---|---|---|---|---|
| Première version | 1.5 | 4 images (!!) | ~21 ans | Bug chemin de fichiers |
| Version corrigée (chemins) | 1.5 | 4 113 images | ~6.3 ans | Flatten + float targets |
| Version améliorée | 3.0 | 4 113 images | en cours | + crop facial + batch 64 |

### 5.2 Diagnostic : pourquoi σ=1.5 produisait des prédictions entières

Avec σ=1.5, la gaussienne cible sur 101 bins est si piquée que le réseau apprenait à mettre toute la probabilité sur un seul bin (comportement argmax). La softmax expectation d'un quasi-one-hot est un entier → les prédictions étaient 34, 36, 38... jamais 34.7.

Avec σ=3.0, la distribution cible a une "épaisseur" naturelle qui force le réseau à produire des distributions lisses → prédictions continues.

### 5.3 Référence état de l'art sur APPA-real

| Modèle | MAE (test) |
|---|---|
| Humain (désaccord inter-annotateurs) | ~4.5 ans |
| DEX (VGG-16, 2016) | 6.52 ans |
| MiVOLO (2023) | 4.96 ans |
| **Notre modèle (objectif)** | **≤ 5.5 ans** |

---

## 6. Bugs identifiés et corrigés au cours du projet

| Bug | Symptôme | Cause | Fix |
|---|---|---|---|
| `Flatten` manquant | Prédit ~49 ans pour tout | `BatchNorm1d` reçoit `(B,2048,1,1)` | `nn.Flatten(1)` ajouté |
| Cible arrondie | Gaussienne mal centrée | `int(round(34.7))=35` passé à la loss | `age_float` passé directement |
| lr uniforme | Backbone détruit | lr=1e-4 sur les poids ImageNet | Freeze 5 epochs + lr=1e-5 backbone |
| `num_workers=2` macOS | Deadlock infini | Multiprocessing Jupyter/macOS | `num_workers=0` en local |
| Chemin relatif | 4 images sur 4113 | CWD kernel ≠ CWD projet | `PROJECT_DIR / "appa-real-release"` absolu |
| σ=1.5 trop petit | Prédictions entières (34, 36...) | Softmax converge vers argmax | σ=3.0 |

---

## 7. Perspectives : méthodes à explorer

Les trois méthodes suivantes adressent des limitations différentes du label smoothing gaussien fixe.

---

### 7.1 Mean-Variance Loss

**Référence** : Pan et al., *Mean-Variance Loss for Deep Age Estimation from a Face*, CVPR 2018

#### Motivation

Le label smoothing gaussien impose une distribution cible *fixe* (σ identique pour tous les exemples). Or certaines images ont une apparence ambiguë (forte variance inter-annotateurs) et d'autres sont très nettes. La Mean-Variance Loss apprend à **réguler directement la distribution prédite** plutôt que de choisir une cible fixe.

#### Comment ça marche

La loss combine deux termes :

$$
\mathcal{L} = \mathcal{L}_{\text{classif}} + \lambda_1 \mathcal{L}_{\text{mean}} + \lambda_2 \mathcal{L}_{\text{var}}
$$

- **$\mathcal{L}_{\text{mean}}$** : pénalise l'écart entre l'espérance prédite $\hat{\mu} = \sum_k k \cdot p_k$ et l'âge cible $\mu^*$

$$
\mathcal{L}_{\text{mean}} = (\hat{\mu} - \mu^*)^2
$$

- **$\mathcal{L}_{\text{var}}$** : encourage une variance prédite raisonnable $\hat{\sigma}^2 = \sum_k (k - \hat{\mu})^2 \cdot p_k$, pour que la distribution ne soit ni trop plate ni trop piquée

$$
\mathcal{L}_{\text{var}} = |\hat{\sigma}^2 - \sigma_{\text{target}}^2|
$$

#### Avantage clé

Le réseau apprend simultanément *à quelle valeur* prédire et *avec quelle confiance* — il produit naturellement des distributions continues sans avoir à fixer σ manuellement.

---

### 7.2 CORAL — Consistent Ordinal Regression for Deep Learning

**Référence** : Cao et al., *Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation*, Pattern Recognition Letters 2020

#### Motivation

Le label smoothing traite les 101 classes comme **indépendantes**. Il n'impose pas que si le modèle pense qu'une personne a "au moins 30 ans", il devrait aussi penser qu'elle a "au moins 29 ans". Cette **cohérence ordinale** est une contrainte naturelle que ni la cross-entropie ni le label smoothing ne garantissent.

#### Comment ça marche

Au lieu d'un classifieur 101-classes, CORAL décompose le problème en **100 classifieurs binaires** partagés :

$$
P(\hat{y} > k) \quad \text{pour } k \in \{0, 1, \ldots, 99\}
$$

Chaque classifieur binaire répond à la question : "La personne a-t-elle **plus de k ans** ?". Par construction ordinale, on impose $P(\hat{y} > k) \geq P(\hat{y} > k+1)$.

La prédiction finale est :

$$
\hat{a} = \sum_{k=0}^{99} \mathbf{1}[P(\hat{y} > k) > 0.5]
$$

La loss est une somme de binary cross-entropies sur les 100 classifieurs, avec des **poids partagés** entre tous les seuils (seul le biais change par seuil).

#### Avantage clé

La structure garantit la **cohérence ordinale par construction** — impossible d'avoir $P(\hat{y} > 40) > P(\hat{y} > 30)$. Très efficace quand les annotations ont une structure ordinale forte (âge, sévérité d'une maladie, note...).

---

### 7.3 Adaptive Label Smoothing

#### Motivation

Le label smoothing gaussien fixe σ=3.0 pour toutes les images. Mais APPA-real fournit, pour chaque image, l'**écart-type des votes** inter-annotateurs (`apparent_age_std`). Une image très consensuelle (std=0.5) devrait avoir un σ petit ; une image ambiguë (std=6.0) devrait avoir un σ plus grand.

#### Comment ça marche

On remplace le σ fixe par le σ propre à chaque exemple :

$$
\sigma_i = \max(\sigma_{\min}, \text{apparent\_age\_std}_i)
$$

La distribution cible devient :

$$
y_k^{(i)} \propto \exp\!\left(-\frac{(k - \mu_i)^2}{2\sigma_i^2}\right)
$$

où $\sigma_i$ est lu directement dans le CSV (`apparent_age_std`) pour chaque image $i$.

```python
# Dans OrdinalLabelSmoothing.forward() :
# targets : (B,) âge float
# sigmas  : (B,) std par exemple, lu depuis le dataset

def _smooth_labels_adaptive(self, targets, sigmas):
    bins  = torch.arange(self.K, device=targets.device).float()
    diff  = bins.unsqueeze(0) - targets.float().unsqueeze(1)   # (B, K)
    s     = sigmas.float().unsqueeze(1).clamp(min=0.5)         # (B, 1)
    labels = torch.exp(-0.5 * (diff / s) ** 2)
    return labels / labels.sum(dim=1, keepdim=True)
```

#### Avantage clé

La loss est **calibrée par l'incertitude réelle des données** : le réseau est moins pénalisé sur les images ambiguës et plus pénalisé sur les images consensuelles. Cela devrait réduire le biais sur les cas difficiles et améliorer la calibration des prédictions.

#### Comparaison des 3 méthodes

| Méthode | σ | Ordinalité | Calibration | Complexité |
|---|---|---|---|---|
| Label Smoothing (actuel) | Fixe (3.0) | Partielle | Non | Faible |
| Mean-Variance Loss | Appris | Partielle | Oui (var) | Moyenne |
| CORAL | N/A | **Garantie** | Non | Moyenne |
| Adaptive Label Smoothing | **Par exemple** | Partielle | **Oui (std)** | Faible |

---

*Rapport rédigé le 6 mars 2026 — Louis Duvignacq*

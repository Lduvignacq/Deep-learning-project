import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils


def get_model_regression(model_name="se_resnext50_32x4d", pretrained="imagenet", hidden_dim=256):
    """
    Modèle pour la régression d'âge (1 sortie continue)
    
    Architecture:
    - 2048 features → hidden_dim (256) → 1 (régression)
    - Avec ReLU + Dropout pour éviter l'overfitting
    """
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features  # 2048
    
    # Remplacer last_linear par un MLP à 2 couches
    model.last_linear = nn.Sequential(
        nn.Linear(dim_feats, hidden_dim),  # 2048 → 256
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(hidden_dim, 1)  # 256 → 1 (sortie continue)
    )
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model


def main():
    model = get_model_regression()
    print(model)


if __name__ == '__main__':
    main()

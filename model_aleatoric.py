import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils

def get_model_aleatoric(model_name="se_resnext50_32x4d", pretrained="imagenet", hidden_dim=256):
    """
    Modèle pour la régression d'âge avec incertitude aléatoire.
    Sorties : [moyenne, log_variance]
    """
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features  # Généralement 2048 pour ResNext50
    
    # On garde ton architecture MLP mais on passe à 2 sorties
    model.last_linear = nn.Sequential(
        nn.Linear(dim_feats, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(hidden_dim, 2)  # [0]: Age, [1]: Log-variance (ou log-scale pour Laplace)
    )
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model

if __name__ == '__main__':
    model = get_model_aleatoric()
    print("Modèle chargé avec 2 sorties pour l'incertitude hétéroscédastique.")
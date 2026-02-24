import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils


def get_model(model_name="se_resnext50_32x4d", pretrained="imagenet"):
    """
    Create a regression model for age estimation.
    Outputs a single continuous value (age) instead of class probabilities.
    """
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    
    # Replace last layer with single output for regression
    model.last_linear = nn.Linear(dim_feats, 1)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    return model


def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()

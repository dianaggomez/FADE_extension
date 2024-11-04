import torch
import torch.nn as nn
import torch.nn.functional as F

from Project.FADE_extension.src.claire_cf_data_augmentation import VAE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulation():
    """ Generate counterfactuals via simulation """
    
    pass

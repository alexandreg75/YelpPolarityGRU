"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> callable (ou None)
"""

def get_augmentation_transforms(config: dict):
    # Pour ce projet NLP, on ne fait pas d'augmentation (baseline propre).
    return None

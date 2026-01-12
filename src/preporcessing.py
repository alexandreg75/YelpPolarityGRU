"""
Pré-traitements pour le NLP (Yelp Polarity).

Signature imposée :
get_preprocess_transforms(config: dict) -> callable
"""

import re
from typing import List, Callable


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenisation simple :
    - minuscules
    - mots alphanumériques
    - conserve les apostrophes (don't, it's, etc.)
    """
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def get_preprocess_transforms(config: dict) -> Callable[[str], List[str]]:
    """
    Retourne une fonction de pré-traitement pour le texte.

    Entrée :
        text (str) : critique Yelp brute
    Sortie :
        tokens (List[str]) : liste de tokens
    """
    def preprocess(text: str) -> List[str]:
        return simple_tokenize(text)

    return preprocess

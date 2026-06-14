# utils.py
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed=42):
    """
    Fixe la 'seed' (graine aléatoire) pour que les résultats soient les mêmes 
    à chaque fois que l'on relance le code (reproductibilité).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device():
    """
    Retourne le meilleur périphérique disponible : CUDA, MPS (pour Apple Silicon M1/M2/M3/M4) ou CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def compute_metrics(predictions, labels):
    """
    Calcule l'Accuracy et le F1-score à partir des prédictions du modèle 
    et des vrais labels.
    """
    # L'accuracy est le pourcentage de bonnes réponses
    acc = accuracy_score(labels, predictions)
    
    # Le F1-score prend en compte les faux positifs et faux négatifs (utile si les classes sont déséquilibrées)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {"accuracy": acc, "f1_score": f1}
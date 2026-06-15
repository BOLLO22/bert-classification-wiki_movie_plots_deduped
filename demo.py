# demo.py
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import create_model

# 1. Chargement des outils nécessaires
device = torch.device("cpu") # Pour tester, le CPU est suffisant
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# On recharge les classes sauvegardées pendant l'entraînement
try:
    classes = torch.load("checkpoints/classes.pt", weights_only=False)
    num_labels = len(classes)
except FileNotFoundError:
    print("Erreur : Entraînez le modèle d'abord (train.py) pour générer les classes.")
    exit()

# 2. Chargement du modèle entraîné
model = create_model(num_labels=num_labels, model_name=model_name)
try:
    # On charge les poids sauvegardés
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval() # Mode évaluation très important pour les prédictions
except FileNotFoundError:
    print("Erreur : Le fichier checkpoints/best_model.pt est introuvable.  train.py d'abord.")
    exit()

# 3. Fonction de prédiction
def predict_tweet(text):
    """Prend un texte en entrée et retourne un dictionnaire de probabilités pour Gradio."""
    # Tokenisation du texte de l'utilisateur
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    # Désactivation des gradients pour la prédiction pure
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        logits = outputs["logits"]
        
        # Application de la fonction Softmax pour transformer les scores (logits) en probabilités (0 à 100%)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
    
    # Création du dictionnaire { "Nom de la classe": probabilité } pour Gradio
    result = {}
    for i, class_name in enumerate(classes):
        result[class_name] = float(probabilities[i])
        
    return result

# 4. Création de l'interface Gradio
interface = gr.Interface(
    fn=predict_tweet,                     # La fonction appelée quand on clique sur "Submit"
    inputs=gr.Textbox(lines=3, placeholder="Entrez un tweet ici..."), # Boîte de texte d'entrée
    outputs=gr.Label(num_top_classes=2),  # Affichage des meilleures probabilités
    title="Classification de Tweets avec BERT",
    description="Cette application utilise un modèle BERT fine-tune pour classifier le type d'un tweet.",
    examples=[
        ["My Husband Beats Me Frequently, Wife Tells Court."],
        ["The government announced a new economic reform today."]
    ] # Des exemples pré-remplis pour faciliter les tests
)

# Lancement de l'application
if __name__ == "__main__":
    interface.launch()
import os
#  POUR MAC 
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # La variable magique pour beaucoup de blocages Mac

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch
# On limite PyTorch à 1 seul thread pour éviter qu'il s'emmêle les pinceaux
torch.set_num_threads(1) 

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from dataset import tweetTypeDataset
from model import create_model
from utils import set_seed, compute_metrics, get_device



def main():
    # 1. Configuration initiale
    set_seed(42)
    device = get_device()
    print(f"Utilisation de l'appareil : {device}")
    
    # Création du dossier pour sauvegarder le modèle
    os.makedirs("checkpoints", exist_ok=True)

    # 2. Chargement des données
    # On charge le CSV avec pandas
    df = pd.read_csv("data/Train.csv")
    df = df.dropna(subset=["tweet", "type"]) # On enlève les lignes vides

    # On encode les labels textuels (le type du tweet) en nombres (0, 1, 2...)
    label_encoder = LabelEncoder()
    df["type_encoded"] = label_encoder.fit_transform(df["type"])
    num_labels = len(label_encoder.classes_)
    
    # Sauvegarde des classes pour la démo Gradio plus tard
    torch.save(label_encoder.classes_, "checkpoints/classes.pt")

    # Split train/validation (80% / 20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["type_encoded"])

    # Calcul des poids des classes pour gérer le déséquilibre
    classes = np.unique(train_df["type_encoded"])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_df["type_encoded"])
    class_weights_tensor = torch.tensor(weights, dtype=torch.float)
    print(f"Poids des classes calculés : {class_weights_tensor}")

    # 3. Préparation du Tokenizer et des Datasets
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_dataset = tweetTypeDataset(train_df, tokenizer, max_length=128, text_column="tweet", label_column="type")
    val_dataset = tweetTypeDataset(val_df, tokenizer, max_length=128, text_column="tweet", label_column="type")

    # Dataloaders pour regrouper les données par lots (batches)
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialisation du modèle et de l'optimiseur
    model = create_model(num_labels=num_labels, model_name=model_name, class_weights=class_weights_tensor)
    model.to(device) # On envoie le modèle sur la carte graphique si disponible

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 3

    best_val_loss = float('inf')

    # 5. Boucle d'entraînement
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch+1} / {epochs} ========")
        
        # --- MODE ENTRAINEMENT ---
        model.train()
        train_loss = 0
        train_preds, train_true = [], []
        
        for batch in tqdm(train_dataloader, desc="Entraînement"):
            # On envoie les données sur le GPU/CPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # On remet les gradients à zéro
            optimizer.zero_grad()

            # Passage dans le modèle (forward)
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            # Rétropropagation (backward)
            loss.backward()
            optimizer.step()

            # Enregistrement des statistiques
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_true.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_dataloader)
        train_metrics = compute_metrics(train_preds, train_true)
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")

        # --- MODE EVALUATION ---
        model.eval() # On désactive certains comportements comme le dropout
        val_loss = 0
        val_preds, val_true = [], []

        # On désactive le calcul des gradients pour gagner de la mémoire et du temps
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
                logits = outputs["logits"]

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        val_metrics = compute_metrics(val_preds, val_true)
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_score']:.4f}")

        # 6. Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            print("Meilleure loss en validation trouvée ! Sauvegarde du modèle...")
            best_val_loss = avg_val_loss
            # On sauvegarde les "poids" du modèle
            torch.save(model.state_dict(), "checkpoints/best_model.pt")

if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class tweetTypeDataset(Dataset):
    """Dataset PyTorch pour classifier la colonne `tweet` suivant `type`."""

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=128,
        text_column="tweet",
        label_column="type",
        label_encoder=None,
    ):
        # Verifie que les colonnes texte et label existent dans le dataframe.
        missing_columns = {
            column
            for column in (text_column, label_column)
            if column not in dataframe.columns
        }
        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes dans le dataset: {sorted(missing_columns)}"
            )

        # Garde seulement les colonnes utiles et supprime les lignes vides.
        self.dataframe = dataframe[[text_column, label_column]].dropna().reset_index(
            drop=True
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

        # Transforme les labels texte de la colonne `type` en nombres.
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(
                self.dataframe[label_column].astype(str)
            )
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(
                self.dataframe[label_column].astype(str)
            )

    def __len__(self):
        # Retourne le nombre total d'exemples dans le dataset.
        return len(self.dataframe)

    def __getitem__(self, index):
        # Recupère un tweet et son label a partir de son index.
        tweet = str(self.dataframe.loc[index, self.text_column])
        label = int(self.labels[index])

        # Tokenise le tweet pour obtenir les entrees attendues par BERT.
        encoding = self.tokenizer(
            tweet,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Retourne les tenseurs utilisés pendant l'entrainement du modele.
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_and_split_data(csv_path="data/Train.csv",
                         text_column="tweet",
                         label_column="type",
                         test_size=0.2,
                         random_state=42):
    """
    Charge le dataset CSV, encode les labels et effectue un split
    train/validation stratifié 80/20.

    """
    df = pd.read_csv(csv_path)

    # Supprime les lignes avec texte ou label manquant
    df = df.dropna(subset=[text_column, label_column])

    texts = df[text_column].astype(str).tolist()
    raw_labels = df[label_column].astype(str).tolist()

    # Encodage des labels (chaînes -> entiers)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_labels)
    num_labels = len(label_encoder.classes_)

    # Split stratifié 80/20
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    return train_texts, val_texts, train_labels, val_labels, label_encoder, num_labels

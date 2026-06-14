import torch
from torch import nn
from transformers import AutoModel


class BerttweetTypeClassifier(nn.Module):
    """Modele BERT pour classifier un tweet selon son type."""

    def __init__(self, num_labels, model_name="bert-base-uncased", dropout=0.3):
        # Initialise la classe parent de PyTorch.
        super().__init__()

        # Charge le modèle BERT pre-entraine choisi.
        self.bert = AutoModel.from_pretrained(model_name)

        # Ajoute du dropout pour réduire le surapprentissage.
        self.dropout = nn.Dropout(dropout)

        # Convertit la sortie de BERT en scores pour chaque classe.
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Envoie les tokens du tweet dans BERT.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Recupère la représentation globale de la phrase.
        pooled_output = outputs.pooler_output

        # Calcule les logits, c'est-a-dire les scores bruts des classes.
        logits = self.classifier(self.dropout(pooled_output))

        # Calcule la loss .
        loss = None
        if labels is not None:
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits, labels)

        # Retourne la loss pour l'entrainement et les logits pour la prediction.
        return {"loss": loss, "logits": logits}


def create_model(num_labels, model_name="bert-base-uncased", dropout=0.3):
    # Cree et retourne une instance du modele de classification.
    return BerttweetTypeClassifier(
        num_labels=num_labels,
        model_name=model_name,
        dropout=dropout,
    )

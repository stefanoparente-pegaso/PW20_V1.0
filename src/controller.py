import os.path
import pathlib
from pathlib import Path

import joblib

from src.DatasetGenerator import generateDataset
from src.dataset_utils import preprocess_dataset
from src.train_models import train_models
from src.evaluate import open_results_dashboard
from src.interface_utils import launch_gradio


# Definizione root, path e creazione cartelle se non presenti
root_path = pathlib.Path(__file__).parent.parent.resolve()
data_dir = root_path / "data"
models_dir = root_path / "trained_models"
data_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

dataset_path = str(data_dir / "dataset.csv")
dataset_bck_path = str(data_dir / "bck")

pkl_paths = {
    'vectorizer_sent_path': str(models_dir / "vectorizer_sent.pkl"),
    'vectorizer_dep_path' : str(models_dir / "vectorizer_dep.pkl"),
    'dep_model_path': str(models_dir / "department_model.pkl"),
    'sent_model_path': str(models_dir / "sentiment_model.pkl")
}


# Costante percentuale recensione per training
training_rows_percentage = 80


def generate_dataset():
    generateDataset(dataset_path, dataset_bck_path)


def view_preprocessed_dataset():
    dataframe = preprocess_dataset(dataset_path, training_rows_percentage, True)
    output_path = Path(dataset_path).parent / "preprocessed_dataset.txt"
    with open(output_path, "w", encoding="utf-8") as file:
        for index, row in dataframe.iterrows():
            riga = f"{row['ID']} - {row['recensione_completa']} - {row['Reparto']} - {row['Sentiment']}\n"
            file.write(riga)
        print(f"Salvataggio completato in: {output_path}")


def train():
    dataframe_80 = preprocess_dataset(dataset_path, training_rows_percentage, True)
    train_models(dataframe_80, pkl_paths)


def check_results():

    dataframe_20 = preprocess_dataset(dataset_path, 100 - training_rows_percentage, False)

    if not os.path.exists(pkl_paths['dep_model_path']) or not os.path.exists(pkl_paths['sent_model_path']):
        print("I modelli non sono ancora stati addestrati. Verrà eseguito anche addestramento")
        train()

    open_results_dashboard(dataframe_20, pkl_paths)


def open_interface():
    if not os.path.exists(pkl_paths['dep_model_path']) or not os.path.exists(pkl_paths['sent_model_path']):
        print("I modelli non sono ancora stati addestrati. Verrà eseguito anche addestramento")
        train()

    vectorizer_dep = joblib.load(pkl_paths['vectorizer_dep_path'])
    vectorizer_sent = joblib.load(pkl_paths['vectorizer_sent_path'])

    model_dep = joblib.load(pkl_paths['dep_model_path'])
    model_sent = joblib.load(pkl_paths['sent_model_path'])

    launch_gradio(vectorizer_dep, vectorizer_sent, model_dep, model_sent)

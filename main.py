import pathlib
from pathlib import Path

from src.DatasetGenerator import generateDataset
from src.load_dataset import preprocess_dataset

training_rows_percentage = 80

def init_config():
    root_path = pathlib.Path(__file__).parent.resolve()
    dataset_path = str(root_path / "data" / "dataset.csv")
    dataset_bck_path = str(root_path / "data" / "bck")
    return dataset_path, dataset_bck_path

# def create_dataset(dataset_path, dataset_bck_path):
#     generateDataset(dataset_path, dataset_bck_path)


def print_menu():
    print()
    print("Le funzionalità disponibili sono le seguenti:")
    print("0. Esci dal programma")
    print("1. Genera un nuovo dataset")
    print("2. Visualizza il dataset corrente dopo l'operazione di preprocessing")
    print("3. Addestra i modelli di ML")
    print("4. Visualizza risultati dei modelli di ML sul dataset fornito")
    print("5. Apri la dashboard interattiva")
    print()


def view_preprocessed_dataset(dataset_path):
    dataframe = preprocess_dataset(dataset_path, training_rows_percentage, True)
    output_path = Path(dataset_path).parent / "preprocessed_dataset.txt"
    with open(output_path, "w", encoding="utf-8") as file:
        for index, row in dataframe.iterrows():
            riga = f"{row['ID']} - {row['recensione_completa']} - {row['Reparto']} - {row['Sentiment']}\n"
            file.write(riga)
        print(f"Salvataggio completato in: {output_path}")


def train(dataset_path):
    dataframe = preprocess_dataset(dataset_path, training_rows_percentage, True)


def main():
    dataset_path, dataset_bck_path = init_config()

    print("Benvenuto nel programma di machine learning PW20.")
    scelta = ""
    while True:
        print_menu()
        scelta = input("Scegli la funzionalità da eseguire: ")
        match scelta:
            case "0": return
            case "1": generateDataset(dataset_path, dataset_bck_path)
            case "2": view_preprocessed_dataset(dataset_path)
            case "3": train(dataset_path)
        print()
        print("====================================================")


if __name__ == "__main__":
    main()

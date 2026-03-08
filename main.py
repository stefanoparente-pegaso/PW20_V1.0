import pathlib
from pathlib import Path

from src.DatasetGenerator import generateDataset
from src.load_dataset import load_data


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


# todo: SE SI CAMBIA LOGICA LOAD_DATA NON RESTITUENDO LISTA DI OGGETTI MA DATAFRAME O ALTRO BISOGNA CAMBIARE LA LOGICA
def view_preprocessed_dataset(dataset_path):
    preprocessed_dataset = load_data(dataset_path)
    output_path = Path(dataset_path).parent / "preprocessed_dataset.txt"
    with open(output_path, "w", encoding="utf-8") as file:
        for row in preprocessed_dataset:
            riga = f"{row['ID']} - {row['testo']} - {row['reparto']} - {row['polarita']}\n"
            file.write(riga)
    print("Dataset preprocessato salvato in " + str(output_path))


def train(dataset_path):
    preprocessed_dataset = load_data(dataset_path)


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

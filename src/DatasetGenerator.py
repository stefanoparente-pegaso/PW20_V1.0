from pathlib import Path
from openai import OpenAI
import os
import shutil
from datetime import datetime

DEPARTMENTS = ["Housekeeping", "Reception", "F&B"]
SENTIMENTS = ["Positivo", "Negativo"]
PROMPT = """
Genera una lista di oggetti JSON con le seguenti chiavi:
- ID (numerico)
- Titolo (stringa)
- Corpo (stringa)
- Reparto ("Housekeeping", "Reception", "F&B")
- Sentiment ("Positivo", "Negativo")

Il contensto della generazione è un dataset per machine larning. Si tratta di un insieme di recensioni di un hotel. Ogni recensione deve essere attribuibile 
a uno dei reparti sopra specificati e deve avere un sentiment positivo o negativo. Nella colonna "Reparto" e "Sentiment" specifica rispettivamente il reparto
rigurdante la recensione e la polarità della stessa. Tali campi serviranno per valutare il modello di ML in fase di test.
Genera un 40% di recensioni positive, un 40% di recensioni negative e un 10% di recensioni dubbie o incerte. Quest'ultima gruppo di recensioni può avere una 
polarità del sentiment non del tutto chiara, un'attribuibilità a più reparti (con uno dominante) o essere completamente senza senso.
Fai in modo che tutte le recensioni siano diverse e che il titolo contenga meno di 10 parole e il corpo meno di 50.
Il numero di recensioni da generare è {n}
"""


def generateReviewsByOpenAi(datasetPath, datasetBckPath):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("KEY OpenAI not defined")
    client = OpenAI(api_key=key)


def generateDataset(datasetPath, datasetBckPath):
    dataset = Path(datasetPath)
    
    # Se il dataset non esiste viene creato
    if not dataset:
        generateReviewsByOpenAi(datasetPath, datasetBckPath)
        print("Un nuovo dataset è stato creato")
        return

    # Se già esiste un dataset si chiede all'utente se vuole sovrascriverlo
    else:
        newDataset = ""
        while (not newDataset):
            newDataset = input("Dataset già esistente, desideri crearne uno nuovo? [S-Y/N]")
            if newDataset.upper() == "S" or newDataset.upper() == "Y":
                dateTime = datetime.today().strftime("%Y%m%d%H%M%S")
                shutil.copytree(datasetPath, datasetBckPath + "/dataset_bck_" + dateTime)
                generateReviewsByOpenAi(datasetPath, datasetBckPath)
                print("Un nuovo dataset è stato creato, il precedente è stato salvato nella cartella di backup")
                return
            else:
                print("Creazione dataset annullata, uscita dal programma...")
                return
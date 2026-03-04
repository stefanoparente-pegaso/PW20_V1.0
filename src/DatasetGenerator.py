from pathlib import Path
from openai import OpenAI
import os
import shutil
from datetime import datetime
import json
import csv

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

def saveReviewsCsv(datasetJson, datasetPath, datasetBckPath):
    print("Salvataggio delle recensioni in formato CSV in corso...")

    # Sostituzione di ';' con ',' in modo da poter usare ';' come separatore per il CSV
    for rec in datasetJson:
        for field in ["Titolo", "Corpo"]:
            if field in rec and isinstance(rec[field], str):
                rec[field] = rec[field].replace(";", ",")

    keys = ["ID", "Titolo", "Corpo", "Reparto", "Sentiment"]

    # Conversione JSON generato da openAI in CSV
    with open(datasetPath, mode='w', newline='', encoding='utf-8') as outputFile:
        dictWriter = csv.DictWriter(outputFile, fieldnames=keys, delimiter=';')
        dictWriter.writeheader()
        dictWriter.writerows(datasetJson)

    print("Dataset salvato")

def generateJsonReviewsByOpenAi():
    print("os.environ:", os.environ.get("OPENAI_API_KEY"))
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("KEY OpenAI not defined")
    client = OpenAI(api_key=key)

    n = 400
    prompt = PROMPT.format(n=n)

    print("Generazione di {n} recensioni in corso...".format(n=n))

    try:
        # Invio propmpt al modello, si richiede una risposta JSON pura
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        responseData = response.choices[0].message.content
        data = json.loads(responseData)

        # Pulizia JSON TODO: C'è bisogna di questa cosa?
        reviews = data if isinstance(data, list) else data.get('recensioni', data.get('data', list(data.values())[0]))

        print("{n} recensioni generate".format(n=n))

        return reviews

    except Exception as e:
        print(e)


def generateDataset(datasetPath, datasetBckPath):

    print (datasetPath)
    # Se il dataset non esiste viene creato
    # TODO Capire se path messo così è corretto (da root) o se bisogna partire da qui
    if not os.path.exists(datasetPath):
        print("Nessun dataset di recensioni presente, si procede con la creazione.")
        datasetJson = generateJsonReviewsByOpenAi()

    # Se già esiste un dataset si chiede all'utente se vuole sovrascriverlo
    else:
        choice = ""
        while True:
            choice = input("Dataset già esistente, desideri crearne uno nuovo? [S-Y/N]").strip().upper()

            if choice in ["S", "Y"]:
                # Backup del precedente datatset nella cartella di backup
                dateTime = datetime.today().strftime("%Y%m%d%H%M%S")
                shutil.copytree(datasetPath, datasetBckPath + "/dataset_bck_" + dateTime)

                print("Un nuovo dataset verrà creato, il precedente è stato salvato nella cartella di backup")
                datasetJson = generateJsonReviewsByOpenAi()
                break

            elif choice in ["N"]:
                print("Creazione dataset annullata, uscita dal programma...")
                return

            else:
                print("Scelta non valida, inserisci un valore tra [S,Y,N]")

    saveReviewsCsv(datasetJson, datasetPath, datasetBckPath)
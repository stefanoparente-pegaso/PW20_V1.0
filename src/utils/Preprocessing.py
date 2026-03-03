import pandas as pd


def preprocessing(datasetPath):

    with open(datasetPath, "w", newline="") as csvfile:
        csvwriter = ""


# TODO Leggere CSV, modificare campi titolo e corpo togliendo spazi, maiuscole, punteggiatura, ecc. Restituire una copia del file csv pulito. Nol modificarlo.
"""
import pandas as pd
df = pd.read_csv("AllDetails.csv")

df.loc[5, 'Name'] = 'SHIV CHANDRA'
df.to_csv("AllDetails.csv", index=False)
print(df)

"""
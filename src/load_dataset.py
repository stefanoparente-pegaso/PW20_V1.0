import pandas as pd
import string


def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def preprocess_dataset(dataset_path, rows_to_return_percentage, ascending):

    df = pd.read_csv(dataset_path, sep=';')
    df['recensione_completa'] = df['Titolo'] + " " + df['Corpo']
    df['recensione_completa'] = df['recensione_completa'].apply(clean_text)
    total_rows = len(df)
    rows_to_return = total_rows * rows_to_return_percentage // 100
    if ascending:
        return df.head(rows_to_return)
    else:
        return df.tail(rows_to_return)


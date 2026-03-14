import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def tokenize_text(rev):
    tokens = word_tokenize(rev) # lista
    stop_words = set(stopwords.words('italian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def clean_text(text):
    text = text.lower().strip().translate(str.maketrans('', '', string.punctuation))
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


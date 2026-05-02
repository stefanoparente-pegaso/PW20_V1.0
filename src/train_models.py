import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.dataset_utils import tokenize_text, embed_dataset

#  LOGISTIC REGRESSION

def create_model(vector, column_df):
    lgr = LogisticRegression(
        random_state=42,
        max_iter=2000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs',
        #multi_class='auto'
    )

    # le categorie della colonna vengono automaticamente convertite in ordine alfabetico
    model = lgr.fit(vector, column_df)
    return model

# LEGENDA:
# random_state = seme casuale
# max_iter = massimo n° iterazioni che algoritmo compie per imparare
# c = regolarizzazione --> più è alto e più si da peso alle singole parole. 1 è una via di mezzo
# solver = algoritmo utilizzato per ottimizzare i pesi
# multi_class = sceglie tra logica 'one vs rest' o multinomial (default auto)



def train_models(dataframe_80, pkl_paths):
    tokens_dep = dataframe_80['recensione_completa'].apply(lambda x: tokenize_text(x, sentiment=False))
    vectorizer_dep = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, use_idf=True)
    rev_vector_dep = embed_dataset(tokens_dep, vectorizer_dep)
    model_dep = create_model(rev_vector_dep, dataframe_80['Reparto'])

    tokens_sent = dataframe_80['recensione_completa'].apply(lambda x: tokenize_text(x, sentiment=True))
    vectorizer_sent = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.7, sublinear_tf=True, use_idf=False)
    rev_vector_sent = embed_dataset(tokens_sent, vectorizer_sent)
    model_sent = create_model(rev_vector_sent, dataframe_80['Sentiment'])

    # Salvataggio vectorizer e modelli
    joblib.dump(vectorizer_dep, pkl_paths['vectorizer_dep_path'])
    joblib.dump(vectorizer_sent, pkl_paths['vectorizer_sent_path'])
    joblib.dump(model_dep, pkl_paths['dep_model_path'])
    joblib.dump(model_sent, pkl_paths['sent_model_path'])



# LEGENDA:
# ngram_range = dimensione sequenze parole da considerare
# min_df = n° minima frequenza della parola nel dataset per essere incula nel dizionario
# max_df = percentuale di frequenza oltre il quale le parole (stop-words) devo essere ignorate
# sublinear_tf = trasformazione logaritmica conteggio, riduce importanza delle parole ripetute più volte nella stessa recensione
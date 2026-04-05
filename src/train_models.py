from sklearn.linear_model import LogisticRegression

def train_model(vector, column_df):
    lgr = LogisticRegression()
    # le categorie della colonna vengono automaticamente convertite in ordine alfabetico
    model = lgr.fit(vector, column_df)
    return model
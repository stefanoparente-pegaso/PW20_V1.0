import os
import webbrowser

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from models.model_result import ModelResult
from .dashboard_html_utils import get_html_model
import pandas as pd

#TODO: aggiungere possibilità di salvare le card html come immagini per metterle nel report?

def show_results(models_to_show):
    dashboard_dir = './static/evaluation_dashboard/'
    template_path = os.path.join(dashboard_dir, 'dashboard.html')
    output_path = os.path.join(dashboard_dir, 'report_generato.html')

    try:

        html_to_replace_list = []

        for model in models_to_show:
            results_html = get_html_model(model)
            html_to_replace_list.append(results_html)

        html_to_replace = ' '.join(html_to_replace_list)
        content = ''

        with open(template_path, 'r') as f:
            content = f.read()

        dashboard = content.replace('%%CARDS_PLACEHOLDER%%', html_to_replace)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dashboard)

        webbrowser.open(f"file://{os.path.abspath(output_path)}")

    except FileNotFoundError:
        print("Errore nel caricamento della dashboard, si mostrano i risultati a terminale:")
        for model in models_to_show:
            print(model.to_dict())

def get_errors_csv(model_name, prediction, column_df, id_df):
    results = pd.DataFrame({
        'ID': id_df,
        'Reale': column_df,
        'Predetto': prediction
    })

    errors = results[results['Reale'] != results['Predetto']]
    target_dir = os.path.join('data', 'prediction_errors')
    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.join(target_dir, f"errori_{model_name}.csv")
    errors.to_csv(filename, index=False, sep=';', encoding='utf-8')

def evaluate_model(name, model, rev_vector, column_df, labels, id_df):

    prediction = model.predict(rev_vector)
    accuracy = accuracy_score(column_df, prediction)
    f1 = f1_score(column_df, prediction, average='macro')
    # conf matrix ordina alfabeticamente F&B, Housekeeping, Reception
    conf_matrix = confusion_matrix(column_df, prediction)

    result = ModelResult(name, accuracy, f1, conf_matrix, labels)
    get_errors_csv(name, prediction, column_df, id_df)


    return result
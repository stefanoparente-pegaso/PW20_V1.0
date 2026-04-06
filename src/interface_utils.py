import gradio as gr
import pandas as pd
from datetime import datetime, timezone
from src.dataset_utils import clean_text, tokenize_text, embed_dataset

def predict(titolo, corpo, vectorizer, model_dep, model_sent):
    recensione = titolo + ' ' + corpo
    recensione = clean_text(recensione)
    tokens = " ".join(tokenize_text(recensione))
    vettore = vectorizer.transform([tokens])
    dep = model_dep.predict(vettore)[0]
    sent = model_sent.predict(vettore)[0]
    # predict_proba restituisce entrambi score x positivo e negativo
    score = float(max(model_sent.predict_proba(vettore)[0]))
    return dep, sent, score

def launch_gradio(vectorizer, model_dep, model_sent):

    with gr.Blocks(title="Classificatore recensioni hotel") as demo:
        gr.Markdown("# Classificatore recensioni hotel")
        gr.Markdown("Inserisci il titolo e il corpo della recensione:")

        def analyze(titolo, corpo, rev_list):
            dep, sent, score = predict(titolo, corpo, vectorizer, model_dep, model_sent)

            # Crea nuova riga per il dataframe
            new_entry = {
                "Titolo": titolo,
                "Corpo": corpo,
                "Reparto": dep,
                "Sentiment": sent,
            }
            updated_data = pd.concat([rev_list, pd.DataFrame([new_entry])], ignore_index=True)
            return dep, sent, score, updated_data

        def delete_selected(data, selected_index: gr.SelectData):
            # selected_index.index[0] ci dice quale riga è stata cliccata
            row_idx = selected_index.index[0]
            updated_data = data.drop(index=row_idx).reset_index(drop=True)
            return updated_data

        def export_to_csv(data):
            if data is None or data.empty:
                return None
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"recensioni_analizzate_{timestamp}.csv"
            data.to_csv(file_path, index=False, sep=';', encoding='utf-8-sig')
            return file_path

        with gr.Row():
            with gr.Column():
                txt_titolo = gr.Textbox(label="Titolo", placeholder="Inserisci qui il titolo")
                txt_corpo = gr.Textbox(label="Testo", placeholder="Inserisci qui la recensione", lines=4)
                btn_run = gr.Button("Analizza Recensione", variant="primary")

            with gr.Column():
                out_dep = gr.Label(label="Reparto Destinatario")
                out_sent = gr.Label(label="Sentiment Rilevato")
                # Usiamo un componente Label o Number per la probabilità richiesta dal PW
                out_score = gr.Number(label="Confidenza Modello (0-1)")

        gr.Markdown("---")
        gr.Markdown("### Lista recensioni analizzate")

        rev_table = gr.Dataframe(
            headers=["Titolo", "Corpo", "Reparto", "Sentiment"],
            datatype=["str", "str", "str", "str"],
            value=pd.DataFrame(columns=["Titolo", "Corpo", "Reparto", "Sentiment"]),
            interactive=False
        )

        with gr.Row():
            btn_export = gr.Button("Esporta in CSV", variant="secondary")
            file_download = gr.File(label="Scarica il file")

        btn_run.click(
            fn=analyze,
            inputs=[txt_titolo, txt_corpo, rev_table],
            outputs=[out_dep, out_sent, out_score, rev_table]
        )

        rev_table.select(
            fn=delete_selected,
            inputs=[rev_table],
            outputs=[rev_table]
        )

        btn_export.click(
            fn=export_to_csv,
            inputs=[rev_table],
            outputs=[file_download]
        )

    demo.launch(inbrowser=True)

    return
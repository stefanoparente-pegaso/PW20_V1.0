def get_html_model(model) :

    html = f'''
    <div class="card">
                <h2 class="card-title">{model.name}</h2>
                
                <div class="stats-container">
                    <div class="stat-item">
                        <div class="stat-value">{model.accuracy}</div>
                        <div class="stat-label">Accuracy</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{model.f1}</div>
                        <div class="stat-label">F1-Score Macro</div>
                    </div>
                </div>

                <h3 class="matrix-title">Confusion Matrix (previsioni vs realtà)</h3>
                {format_matrix_to_html(model.conf_matrix, model.labels)}
                
            </div>
    '''

    return html


def format_matrix_to_html(matrix, labels):

    html = "<table class='matrix-table'>"

    # html += "<tr><th>Reale \ Predetto</th>"
    for label in labels:
        html += f"<th>{label}</th>"
    html += "</tr>"

    for i, row in enumerate(matrix):
        html += "<tr>"
        for j, val in enumerate(row):
            # Se siamo sulla diagonale principale (i == j), è una previsione CORRETTA
            css_class = "matrix-correct" if i == j else "matrix-error"

            # Non mostriamo le classi 'error' se il valore è zero
            if val == 0 and i != j:
                css_class = "matrix-cell"

            html += f"<td class='{css_class}'>{val}</td>"
        html += "</tr>"
    html += "</table>"

    return html


import base64
from io import BytesIO
import os
import pickle

import numpy as np
try:
    from tf_keras.models import load_model, Model
    print("Using tf_keras for model loading.")
except ImportError:
    from tensorflow.keras.models import load_model, Model
    print("Using tensorflow.keras for model loading.")

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

from odontogenic_helpers import parse_image, create_img, label_mapping, load_tsne_data

# 1. Load Data and Model
print("Loading T-SNE data...")
all_X_hat, all_labels, image_paths, linear_model = load_tsne_data()

MODEL_PATH = "Resultado_dataset-antigo/Treinamento_ADAM_InceptionV3_03-05-2026__18-08-53/models/best_modelFold3.keras"
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# Feature extractor for mapping new images to T-SNE
try:
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
except:
    # Fallback if layer structure is different
    feature_layer_name = None
    for layer in model.layers:
        if "global_max_pooling2d" in layer.name:
            feature_layer_name = layer.name
            break
    if feature_layer_name:
        feature_model = Model(inputs=model.input, outputs=model.get_layer(feature_layer_name).output)
    else:
        feature_model = model # Fallback to full model (might not work for linear model mapping)

intro_text = """
Este aplicativo aplica T-SNE nas imagens do dataset de cistos odontogênicos (Queratocisto vs Dentígero),
reduzindo cada imagem a uma incorporação bidimensional para visualizar a semelhança entre elas.
Clusters representam imagens semelhantes. O aplicativo também permite prever a classe de cada imagem
usando uma rede neural convolucional InceptionV3 pré-treinada.

Passe o mouse sobre cada ponto no gráfico T-SNE para ver a imagem que ele representa.
Você pode clicar em um ponto individual para ver a previsão da CNN para esse ponto,
bem como o rótulo real. Você também pode carregar sua própria imagem para ver como a CNN a classificaria.
"""

def create_tsne_graph(uploaded_point=None):
    colors = px.colors.qualitative.Set1 # Using Set1 for distinct colors
    traces = []
    
    for i, label_name in label_mapping.items():
        idx = np.where(all_labels == i)[0]
        x = all_X_hat[idx, 0]
        y = all_X_hat[idx, 1]
        
        trace = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(color=colors[i % len(colors)], size=5),
            name=label_name,
            text=[f"Path: {os.path.basename(image_paths[j])}" for j in idx],
            customdata=idx,
            opacity=0.8,
            showlegend=True
        )
        traces.append(trace)

    annotation = []
    if uploaded_point is not None:
        annotation.append(
            dict(
                x=uploaded_point[0][0],
                y=uploaded_point[0][1],
                xref="x",
                yref="y",
                text="Imagem Carregada",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-40,
                font=dict(size=16, color="black"),
                bgcolor="white",
                bordercolor="black"
            )
        )

    layout = go.Layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        clickmode='event+select',
        annotations=annotation,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return go.Figure(data=traces, layout=layout)

app = dash.Dash(__name__, external_scripts=['https://cdn.plot.ly/plotly-latest.min.js'])
app.title = "Odontogenic Cyst Explorer"
server = app.server

# --- Layout ---
app.layout = html.Div(
    style={'fontFamily': 'sans-serif', 'padding': '20px'},
    children=[
        html.Div(
            id="app-header",
            style={'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '15px', 'marginBottom': '20px', 'borderRadius': '5px'},
            children=[
                html.H2("Explorador de Cistos Odontogênicos: T-SNE e CNN", style={'margin': '0'}),
            ],
        ),
        
        html.Details(
            style={'marginBottom': '20px', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'},
            children=[
                html.Summary(html.B("Sobre este Aplicativo"), style={'cursor': 'pointer'}),
                dcc.Markdown(intro_text)
            ],
        ),

        html.Div(
            style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'},
            children=[
                # Sidebar / Control Panel
                html.Div(
                    id="control-card",
                    style={'flex': '1', 'minWidth': '300px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'},
                    children=[
                        html.H4("Controles"),
                        html.P("Carregar uma imagem para classificação:"),
                        dcc.Upload(
                            id="img-upload",
                            style={
                                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                'textAlign': 'center', 'margin': '10px 0'
                            },
                            children=html.Div(["Arraste e Solte ou ", html.A("Selecione")]),
                        ),
                        html.Div(id="output-img-upload"),
                    ],
                ),
                
                # Main Graph Area
                html.Div(
                    style={'flex': '3', 'minWidth': '600px'},
                    children=[
                        html.Div(
                            style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'},
                            children=[
                                html.H3("Visualização T-SNE das Imagens", style={'marginTop': '0'}),
                                dcc.Graph(id="tsne-graph", figure=create_tsne_graph()),
                            ]
                        ),
                        
                        # Preview Cards
                        html.Div(
                            style={'display': 'flex', 'gap': '20px', 'marginTop': '20px'},
                            children=[
                                # Hover Preview
                                html.Div(
                                    className="img-card",
                                    style={'flex': '1', 'padding': '15px', 'backgroundColor': 'white', 'borderRadius': '5px', 'border': '1px solid #eee', 'textAlign': 'center'},
                                    children=[
                                        html.B("Imagem ao Passar o Mouse:"),
                                        html.Br(), html.Br(),
                                        html.Img(id="hover-point-graph", style={'maxWidth': '100%', 'height': 'auto', 'maxHeight': '300px', 'border': '1px solid #ccc'}),
                                    ],
                                ),
                                # Selection/Prediction Results
                                html.Div(
                                    className="img-card",
                                    style={'flex': '1', 'padding': '15px', 'backgroundColor': 'white', 'borderRadius': '5px', 'border': '1px solid #eee', 'textAlign': 'center'},
                                    children=[
                                        html.B("Ponto Selecionado:"),
                                        html.Div(id="prediction", children=[
                                            html.P("Clique em um ponto para ver a previsão da rede.")
                                        ]),
                                        html.Img(id="selected-data-graph", style={'maxWidth': '100%', 'height': 'auto', 'maxHeight': '300px', 'border': '1px solid #ccc'}),
                                    ],
                                ),
                            ]
                        )
                    ],
                ),
            ],
        ),
    ]
)

# --- Callbacks ---

@app.callback(
    [Output("output-img-upload", "children"), Output("tsne-graph", "figure")],
    [Input("img-upload", "contents")],
    [State("img-upload", "filename"), State("img-upload", "last_modified")]
)
def handle_upload(contents, fname, date):
    if contents is None:
        return None, create_tsne_graph()
        
    orig_img_b64, resized_arr = parse_image(contents, fname, date)
    
    # Predict Class
    img_batch = np.expand_dims(resized_arr, axis=0)
    pred_arr = model.predict(img_batch)
    pred_idx = np.argmax(pred_arr)
    prob = np.max(pred_arr) * 100
    
    # Predict T-SNE Position
    feat = feature_model.predict(img_batch).flatten().reshape(1, -1)
    tsne_pos = linear_model.predict(feat)
    
    result_div = html.Div([
        html.P(f"Arquivo: {fname}"),
        html.Img(src=orig_img_b64, style={'maxWidth': '100%', 'maxHeight': '200px', 'borderRadius': '5px'}),
        html.H5(f"Previsão: {label_mapping[pred_idx]}"),
        html.P(f"Certeza: {prob:.2f}%"),
        html.Button("Remover", id="clear-upload", n_clicks=0, style={'marginTop': '10px'})
    ])
    
    return result_div, create_tsne_graph(uploaded_point=tsne_pos)

@app.callback(
    Output("hover-point-graph", "src"),
    [Input("tsne-graph", "hoverData")]
)
def update_hover(hoverData):
    if not hoverData:
        # Default to first image
        return create_img(image_paths[0])
    
    idx = hoverData["points"][0]["customdata"]
    return create_img(image_paths[idx])

@app.callback(
    [Output("selected-data-graph", "src"), Output("prediction", "children")],
    [Input("tsne-graph", "clickData")]
)
def update_click(clickData):
    if not clickData:
        return "", html.P("Clique em um ponto para ver a previsão da rede.")
    
    idx = clickData["points"][0]["customdata"]
    path = image_paths[idx]
    
    # Actual label
    actual_label = label_mapping[all_labels[idx]]
    
    # Model prediction for this image
    # (In a real app we might precompute this, but here we do it on the fly for simplicity)
    from PIL import Image
    im = Image.open(path).convert('RGB').resize((299, 299))
    arr = np.array(im) / 255.0
    pred_arr = model.predict(np.expand_dims(arr, axis=0))
    pred_idx = np.argmax(pred_arr)
    prob = np.max(pred_arr) * 100
    
    color = "green" if pred_idx == all_labels[idx] else "red"
    
    prediction_info = [
        html.H5(f"Previsão: {label_mapping[pred_idx]}", style={'color': color, 'margin': '5px 0'}),
        html.P(f"Certeza: {prob:.2f}%"),
        html.P(f"Real: {actual_label}", style={'fontWeight': 'bold'})
    ]
    
    return create_img(path), prediction_info

if __name__ == "__main__":
    app.run(debug=False, port=8051)

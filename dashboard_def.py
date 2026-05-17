import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import glob
import base64
from io import BytesIO

# ──────────────────────────────────────────────────────────────────────────────
# Configurações de paths
# ──────────────────────────────────────────────────────────────────────────────
DATASET_DIM_PATH = "dimensoes-dataset"
RESULTS_DIM = {
    "Antigo": "datasets/Resultado_dataset-antigo",
    "Novo":   "datasets/Resultado_dataset-novo",
}


# ──────────────────────────────────────────────────────────────────────────────
# Paleta de cores e estilos DDK
# ──────────────────────────────────────────────────────────────────────────────
PLUM         = "#7C2D5A"
PLUM_L       = "#A63D72"
PAGE_BG      = "#f4f4f6"
CARD_BG      = "#FFFFFF"
TEXT         = "#2C2C2C"
MUTED        = "#6b6b6b"
HANDLE_COLOR = "rgb(139,75,107)"
COLORWAY     = [
    "rgb(18,97,228)", "rgb(252,124,76)", "rgb(62,204,252)",
    "rgb(155,200,238)", "rgb(252,138,103)", "rgb(252,173,146)",
    "rgb(139,75,107)", "rgb(252,192,174)",
]

FONT_SANS = "'DM Sans',Arial,sans-serif"
FONT_MONO = "'DM Mono',monospace"

card_style = {
    "background": CARD_BG,
    "border": "1px solid #E8E8E8",
    "borderRadius": "8px",
    "boxShadow": "0px 2px 8px rgba(139,75,107,0.10)",
    "outlineWidth": "1px",
    "outlineStyle": "solid",
    "outlineColor": "#E8E8E8",
    "boxSizing": "border-box",
}

LAYOUT_COMMON = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FAFAFA",
    font=dict(family=FONT_SANS, color=TEXT, size=12),
    title_font=dict(family=FONT_SANS, color=TEXT, size=14),
    margin=dict(l=40, r=20, t=44, b=40),
    colorway=COLORWAY,
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#E8E8E8", borderwidth=1),
    xaxis=dict(gridcolor="#F0F0F0", linecolor="#E8E8E8", zerolinecolor="#E8E8E8"),
    yaxis=dict(gridcolor="#F0F0F0", linecolor="#E8E8E8", zerolinecolor="#E8E8E8"),
)

TAB_STYLE = {
    "borderTop": "none",
    "borderBottom": "3px solid transparent",
    "color": MUTED,
    "fontFamily": FONT_SANS,
    "padding": "12px 16px",
    "fontWeight": "400",
    "background": PAGE_BG,
}
TAB_SELECTED_STYLE = {
    "borderTop": "none",
    "borderBottom": f"3px solid {HANDLE_COLOR}",
    "color": HANDLE_COLOR,
    "fontFamily": FONT_SANS,
    "padding": "12px 16px",
    "fontWeight": "600",
    "background": CARD_BG,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers de UI
# ──────────────────────────────────────────────────────────────────────────────

def kpi_card(title, value):
    """DDK DataCard com Handle bar lateral direito."""
    return html.Div(
        html.Div([
            html.Div([
                html.Div(value, style={
                    "fontSize": "26px", "fontWeight": "700",
                    "lineHeight": "1.2", "color": TEXT,
                    "fontFamily": FONT_SANS,
                }),
                html.Div(title, style={
                    "fontSize": "12px", "color": MUTED,
                    "marginTop": "6px", "fontFamily": FONT_SANS,
                }),
            ], style={"padding": "16px", "backgroundColor": CARD_BG, "flex": "1"}),
            html.Div(style={
                "backgroundColor": HANDLE_COLOR,
                "borderRadius": "0px calc(5.33333px) calc(5.33333px) 0px",
                "width": "12px", "minWidth": "12px", "alignSelf": "stretch",
            }),
        ], style={
            **card_style,
            "display": "flex", "flexDirection": "row",
            "alignItems": "stretch", "overflow": "hidden",
            "flex": "1", "minWidth": "160px",
        }),
        style={"flex": "1", "minWidth": "160px"},
    )


def make_kpi_row(df):
    """Gera a lista de kpi_cards baseada num DataFrame."""
    if df.empty:
        return [kpi_card("Total", "0"), kpi_card("Classes", "0"),
                kpi_card("Largura Média", "0"), kpi_card("Altura Média", "0")]
    
    n_imgs   = len(df)
    n_class  = df["directory"].nunique() if "directory" in df.columns else "—"
    avg_w    = df["width"].mean()  if "width"  in df.columns else 0
    avg_h    = df["height"].mean() if "height" in df.columns else 0

    return [
        kpi_card("Total de Imagens",   f"{n_imgs:,}"),
        kpi_card("Classes",            str(n_class)),
        kpi_card("Largura Média (px)", f"{avg_w:.0f}"),
        kpi_card("Altura Média (px)",  f"{avg_h:.0f}"),
    ]


def section_title(text):
    return html.Div(text, style={
        "fontSize": "15px", "fontWeight": "600", "color": TEXT,
        "fontFamily": FONT_SANS, "marginBottom": "12px", "marginTop": "8px",
    })


def graph_card(fig, height=320, id_=None):
    if id_ is None:
        import uuid
        id_ = f"graph-{str(uuid.uuid4())[:8]}"
    return html.Div(
        dcc.Graph(id=id_, figure=fig, config={"displayModeBar": False},
                  style={"height": f"{height}px"}),
        style={**card_style, "flex": "1", "minWidth": "45%", "padding": "8px"},
    )


def two_col(*figs, height=320, id_prefix=""):
    cards = [graph_card(f, height, id_=f"{id_prefix}-graph-{i}") for i, f in enumerate(figs)]
    rows = []
    for i in range(0, len(cards), 2):
        rows.append(html.Div(cards[i:i+2],
                             style={"display": "flex", "gap": "16px",
                                    "flexWrap": "wrap", "marginBottom": "16px"}))
    return html.Div(rows)


def label_badge(text, color=PLUM):
    return html.Span(text, style={
        "background": color, "color": "#fff", "borderRadius": "20px",
        "padding": "3px 10px", "fontSize": "11px", "fontFamily": FONT_MONO,
    })


def dropdown(id_, options, value, label=""):
    return html.Div([
        html.Label(label, style={"color": MUTED, "fontSize": "12px",
                                  "fontFamily": FONT_SANS, "marginBottom": "4px"}),
        dcc.Dropdown(
            id=id_, options=options, value=value, clearable=False,
            style={"fontFamily": FONT_SANS, "fontSize": "13px",
                   "border": "1px solid #E8E8E8", "borderRadius": "6px"},
        ),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Dados auxiliares (funções de leitura)
# ──────────────────────────────────────────────────────────────────────────────

def get_training_runs(dataset_type):
    base = RESULTS_DIM.get(dataset_type, "")
    if not base or not os.path.exists(base):
        return []
    runs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    return sorted(runs)


def load_training_data(dataset_type, run_name):
    path = os.path.join(RESULTS_DIM[dataset_type], run_name)
    log_files = glob.glob(os.path.join(path, "log_Fold*.csv"))
    all_data = []
    for f in log_files:
        try:
            fold = os.path.basename(f).replace("log_Fold", "").replace(".csv", "")
            df = pd.read_csv(f, sep=";")
            df["Fold"] = f"Fold {fold}"
            all_data.append(df)
        except Exception as e:
            print(f"Erro ao carregar {f}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def load_dim_csv(dataset_type):
    path = os.path.join(DATASET_DIM_PATH, f"image_dimensions_dataset_{dataset_type}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_run_summary(dataset_type, run_name):
    """Carrega o arquivo results_*.csv se existir na pasta do treino."""
    path = os.path.join(RESULTS_DIM[dataset_type], run_name)
    files = glob.glob(os.path.join(path, "results_*.csv"))
    if not files:
        return pd.DataFrame()
    try:
        # Pega o primeiro que encontrar
        df = pd.read_csv(files[0], sep=";")
        return df
    except Exception as e:
        print(f"Erro ao carregar sumário {files[0]}: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Dashboard de Análise TCC — Cistos Odontogênicos"
server = app.server

# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # ── Header ─────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1(
                "Dashboard de Análise TCC — Cistos Odontogênicos",
                style={"color": "#fff", "fontFamily": FONT_SANS,
                       "fontSize": "clamp(18px,2.5vw,28px)", "fontWeight": "700",
                       "marginBottom": "8px", "lineHeight": "1.3"},
            ),
            html.P(
                "Visualização interativa de datasets, treinamentos e explorador T-SNE "
                "para classificação de cistos Queratocisto vs Dentígero via InceptionV3.",
                style={"color": "rgba(255,255,255,0.80)", "fontSize": "13px",
                       "fontFamily": FONT_SANS, "lineHeight": "1.6", "maxWidth": "700px"},
            ),
            html.Div([
                label_badge("🦷 Odontogenic Cyst Classifier"),
                label_badge("📊 InceptionV3", color="#1261e4"),
                label_badge("🗄 K-Fold Cross Validation", color="#555"),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px",
                      "marginTop": "14px"}),
        ], style={"flex": "1"}),
    ], style={
        "background": f"linear-gradient(135deg,{PLUM} 0%,{PLUM_L} 60%,#3a1a2e 100%)",
        "padding": "32px 40px", "display": "flex", "gap": "24px",
        "alignItems": "flex-start",
    }),

    # ── Body ───────────────────────────────────────────────────────────────────
    html.Div([

        # ── Tabs principais ────────────────────────────────────────────────────
        dcc.Tabs(id="main-tabs", value="tab-datasets", children=[
            dcc.Tab(label="📐 Dimensões do Dataset",    value="tab-datasets",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="🎯 Análise de Treinamento",  value="tab-training",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="🔬 Predição da lesão",           value="tab-tsne",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        ], style={"borderBottom": "1px solid #E8E8E8"},
           colors={"border": "transparent", "primary": "transparent",
                   "background": PAGE_BG}),

        # ── Conteúdo das tabs ──────────────────────────────────────────────────
        html.Div(id="tab-content", style={"paddingTop": "24px"}),

    ], style={"background": PAGE_BG, "padding": "24px 40px",
              "minHeight": "calc(100vh - 180px)"}),

], style={"background": PAGE_BG, "minHeight": "100vh"})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACK: renderiza a tab
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):

    # ── TAB 1: Dimensões do Dataset (LADO A LADO) ──────────────────────────
    if tab == "tab-datasets":
        return html.Div([
            html.Div([
                get_dataset_components("novo"),
                get_dataset_components("antigo"),
            ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
        ])

    # ── TAB 2: Análise de Treinamento ────────────────────────────────────────
    elif tab == "tab-training":
        return html.Div([
            # Filtros
            html.Div([
                html.Div([
                    dropdown("train-ds-sel",
                             [{"label": "Antigo", "value": "Antigo"},
                              {"label": "Novo",   "value": "Novo"}],
                             "Antigo", "Dataset"),
                ], style={"flex": "1", "minWidth": "180px"}),
                html.Div([
                    dropdown("run-sel", [], None, "Treinamento"),
                ], style={"flex": "2", "minWidth": "260px"}),
                html.Div([
                    dropdown("compare-ds-sel",
                             [{"label": "Nenhum",  "value": "none"},
                              {"label": "Antigo",  "value": "Antigo"},
                              {"label": "Novo",    "value": "Novo"}],
                             "none", "Comparar com Dataset"),
                ], style={"flex": "1", "minWidth": "180px"}),
                html.Div([
                    dropdown("compare-run-sel", [{"label":"—","value":"none"}],
                             "none", "Treinamento para comparar"),
                ], style={"flex": "2", "minWidth": "260px"}),
            ], style={**card_style, "padding": "16px", "display": "flex",
                      "flexWrap": "wrap", "gap": "16px", "marginBottom": "20px"}),

            # KPIs de treinamento
            html.Div(id="train-kpis",
                     style={"display": "flex", "flexWrap": "wrap",
                             "gap": "12px", "marginBottom": "20px"}),

            # Gráficos
            html.Div(id="train-charts"),
        ])

    # ── TAB 3: Análise T-SNE ─────────────────────────────────────────────────
    elif tab == "tab-tsne":
        return html.Div([
            html.Div([
                # Coluna esquerda: input + info
                html.Div([
                    html.Div([
                        section_title("📁 Input — Carregar Imagem"),
                        dcc.Upload(
                            id="tsne-upload",
                            children=html.Div([
                                html.Div("⬆", style={"fontSize": "32px",
                                                      "color": HANDLE_COLOR}),
                                html.Div("Drag & Drop ou clique para selecionar",
                                         style={"fontSize": "13px", "color": MUTED,
                                                "fontFamily": FONT_SANS}),
                            ], style={"textAlign": "center", "padding": "20px"}),
                            style={
                                "border": f"2px dashed {HANDLE_COLOR}",
                                "borderRadius": "8px", "cursor": "pointer",
                                "background": "#fdf5f9",
                            },
                            accept="image/*",
                        ),
                        html.Div(id="tsne-upload-info",
                                 style={"marginTop": "12px"}),
                    ], style={**card_style, "padding": "16px",
                              "marginBottom": "16px"}),

                    # Imagem de predição
                    html.Div([
                        section_title("🖼 Imagem de Predição"),
                        html.Div(id="tsne-pred-image",
                                 style={"textAlign": "center", "minHeight": "160px",
                                        "display": "flex", "alignItems": "center",
                                        "justifyContent": "center",
                                        "color": MUTED, "fontSize": "13px",
                                        "fontFamily": FONT_SANS}),
                    ], style={**card_style, "padding": "16px"}),

                ], style={"flex": "1", "minWidth": "280px", "maxWidth": "340px",
                          "display": "flex", "flexDirection": "column"}),

                # Coluna direita: 
                html.Div([
                    # Acurácia + predições
                    html.Div([
                        section_title("📊 Acurácia & Predições"),
                        html.Div(id="tsne-accuracy-area",
                                 style={"display": "flex", "gap": "12px",
                                        "flexWrap": "wrap", "marginBottom": "16px"}),
                        html.Div(id="tsne-pred-table"),
                    ], style={**card_style, "padding": "16px"}),
                    # T-SNE plot + tabela de predições
                    html.Div([
                        section_title("🔬 Análise T-SNE — Modos do Modelo"),
                    html.Div([
                            dropdown("tsne-mode-sel",
                                     [{"label": "T-SNE Completo",    "value": "full"},
                                      {"label": "Por Classe",        "value": "class"},
                                      {"label": "Confiança (alpha)", "value": "conf"}],
                                     "full", "Modo de visualização"),
                        ], style={"width": "220px", "marginBottom": "12px"}),
                        dcc.Graph(id="tsne-graph",
                                  config={"displayModeBar": False},
                                  style={"height": "380px"}),
                    ], style={**card_style, "padding": "16px",
                              "marginBottom": "16px"}),
                ], style={"flex": "3", "minWidth": "400px",
                          "display": "flex", "flexDirection": "column"}),

            ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DIMENSÕES DO DATASET - Transmform data
# ══════════════════════════════════════════════════════════════════════════════

def get_dataset_components(ds_type):

    df = load_dim_csv(ds_type)

    if df.empty:
        return html.Div(
            f"⚠ Arquivo não encontrado: {DATASET_DIM_PATH}/image_dimensions_dataset_{ds_type}.csv",
            style={"color": MUTED, "fontFamily": FONT_SANS, "padding": "20px"},
        )

    # --- Filtros e Estatísticas (Padrão Nomeado) ---
    # Adequação das antigas linhas 407-420
    df_querato = df.query("directory == 'queratocisto'")
    
    stats_querato = {
        "total": len(df_querato),
        "media_largura": df_querato["width"].mean() if not df_querato.empty else 0,
        "media_altura": df_querato["height"].mean() if not df_querato.empty else 0
    }

    n_imgs   = len(df)
    n_class  = df["directory"].nunique() if "directory" in df.columns else "—"
    avg_w    = df["width"].mean()  if "width"  in df.columns else 0
    avg_h    = df["height"].mean() if "height" in df.columns else 0

    kpis = html.Div([
        kpi_card("Total de Imagens",   f"{n_imgs:,}"),
        kpi_card("Classes",            str(n_class)),
        kpi_card("Largura Média (px)", f"{avg_w:.0f}"),
        kpi_card("Altura Média (px)",  f"{avg_h:.0f}"),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "marginBottom": "16px"})

    # Gráficos
    counts = df["directory"].value_counts().reset_index()
    counts.columns = ["Classe", "Quantidade"]
    fig_cls = px.bar(
        counts,
        x="Classe", y="Quantidade",
        title=f"Distribuição de Classes — {ds_type.upper()}",
        color="Classe",
    )
    fig_cls.update_layout(**LAYOUT_COMMON)

    fig_sc = px.scatter(df, x="width", y="height", color="directory",
                        opacity=0.6, title=f"Largura vs Altura — {ds_type.upper()}",
                        hover_data=["filename"] if "filename" in df.columns else None)
    fig_sc.update_layout(**LAYOUT_COMMON)

    fig_hw = px.histogram(df, x="width",  nbins=50,
                          title=f"Distribuição de Largura — {ds_type.upper()}",
                          color_discrete_sequence=["rgb(18,97,228)"])
    fig_hw.update_layout(**LAYOUT_COMMON)

    fig_hh = px.histogram(df, x="height", nbins=50,
                          title=f"Distribuição de Altura — {ds_type.upper()}",
                          color_discrete_sequence=["rgb(252,124,76)"])
    fig_hh.update_layout(**LAYOUT_COMMON)

    # Tabela estatísticas
    stats = df[["width","height"]].describe().reset_index()
    table = html.Div([
        section_title(f"Estatísticas Descritivas — {ds_type.upper()}"),
        dash_table.DataTable(
            data=stats.to_dict("records"),
            columns=[{"name": c, "id": c} for c in stats.columns],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": PLUM, "color": "#fff",
                          "fontWeight": "600", "fontFamily": FONT_SANS,
                          "fontSize": "11px"},
            style_cell={"fontFamily": FONT_SANS, "fontSize": "11px",
                        "padding": "6px", "textAlign": "left"},
            style_data_conditional=[
                {"if": {"row_index": "odd"},
                 "backgroundColor": "#fdf5f9"}
            ],
        ),
    ], style={**card_style, "padding": "12px", "marginBottom": "16px"})

    return html.Div([
        section_title(f"📊 DATASET {ds_type.upper()}"),
        kpis,
        graph_card(fig_cls, 300, id_=f"fig-cls-{ds_type}"),
        html.Div(style={"height": "16px"}),
        graph_card(fig_sc, 300, id_=f"fig-sc-{ds_type}"),
        html.Div(style={"height": "16px"}),
        two_col(fig_hw, fig_hh, height=280, id_prefix=ds_type),
        table,
    ], style={"flex": "1", "minWidth": "400px"})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISE DE TREINAMENTO
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("run-sel", "options"),
    Output("run-sel", "value"),
    Input("train-ds-sel", "value"),
)
def update_run_options(ds):
    runs = get_training_runs(ds)
    opts = [{"label": r, "value": r} for r in runs]
    return opts, (runs[0] if runs else None)


@app.callback(
    Output("compare-run-sel", "options"),
    Output("compare-run-sel", "value"),
    Input("compare-ds-sel", "value"),
)
def update_compare_run(ds):
    if ds == "none":
        return [{"label": "—", "value": "none"}], "none"
    runs = get_training_runs(ds)
    opts = [{"label": "—", "value": "none"}] + [{"label": r, "value": r} for r in runs]
    return opts, "none"


@app.callback(
    Output("train-kpis",   "children"),
    Output("train-charts", "children"),
    Input("train-ds-sel",     "value"),
    Input("run-sel",          "value"),
    Input("compare-ds-sel",   "value"),
    Input("compare-run-sel",  "value"),
)
def update_training(ds, run, cmp_ds, cmp_run):
    if not run:
        return [], html.Div("Selecione um treinamento.",
                            style={"color": MUTED, "fontFamily": FONT_SANS,
                                   "padding": "20px"})

    df = load_training_data(ds, run)
    if df.empty:
        return [], html.Div("Nenhum log CSV encontrado.",
                            style={"color": MUTED, "fontFamily": FONT_SANS,
                                   "padding": "20px"})

    # KPIs
    best = df.groupby("Fold").agg(
        {"val_accuracy": "max", "accuracy": "max",
         "val_loss": "min", "loss": "min"}
    )
    kpis = [
        kpi_card("Melhor Val Accuracy", f"{best['val_accuracy'].max():.4f}"),
        kpi_card("Melhor Accuracy",     f"{best['accuracy'].max():.4f}"),
        kpi_card("Menor Val Loss",      f"{best['val_loss'].min():.4f}"),
        kpi_card("Menor Loss",          f"{best['loss'].min():.4f}"),
        kpi_card("Folds",               str(df['Fold'].nunique())),
    ]

    # Gráficos base
    
    fig_acc = px.line(df, x="epoch", y="accuracy",     color="Fold",
                      title=f"Acurácia Treino — {run}")
    fig_vac = px.line(df, x="epoch", y="val_accuracy", color="Fold",
                      title=f"Acurácia Validação — {run}")
    fig_los = px.line(df, x="epoch", y="loss",         color="Fold",
                      title=f"Loss Treino — {run}")
    fig_vlo = px.line(df, x="epoch", y="val_loss",     color="Fold",
                      title=f"Loss Validação — {run}")

    for f in [fig_acc, fig_vac, fig_los, fig_vlo]:
        f.update_layout(**LAYOUT_COMMON)

    # Comparação (se selecionada)
    compare_block = html.Div()
    if cmp_ds != "none" and cmp_run and cmp_run != "none":
        df2 = load_training_data(cmp_ds, cmp_run)
        if not df2.empty:
            df["source"]  = f"{ds} / {run}"
            df2["source"] = f"{cmp_ds} / {cmp_run}"
            merged = pd.concat([df, df2], ignore_index=True)

            fig_cmp_acc = px.line(merged, x="epoch", y="val_accuracy",
                                  color="source", line_dash="source",
                                  title="Comparação — Val Accuracy")
            fig_cmp_los = px.line(merged, x="epoch", y="val_loss",
                                  color="source", line_dash="source",
                                  title="Comparação — Val Loss")
            fig_cmp_acc.update_layout(**LAYOUT_COMMON)
            fig_cmp_los.update_layout(**LAYOUT_COMMON)

            compare_block = html.Div([
                section_title("⚖ Comparação de Treinamentos"),
                two_col(fig_cmp_acc, fig_cmp_los, id_prefix="compare"),
            ], style={**card_style, "padding": "16px", "marginBottom": "16px"})

    # Tabela melhores métricas (per-fold)
    best_df = best.reset_index().round(4)
    table_folds = html.Div([
        section_title("🏆 Melhores Métricas por Fold (Logs CSV)"),
        dash_table.DataTable(
            data=best_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in best_df.columns],
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": PLUM, "color": "#fff",
                          "fontWeight": "600", "fontFamily": FONT_SANS,
                          "fontSize": "12px"},
            style_cell={"fontFamily": FONT_SANS, "fontSize": "12px",
                        "padding": "8px", "textAlign": "left"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#fdf5f9"}
            ],
        ),
    ], style={**card_style, "padding": "16px", "marginBottom": "16px"})

    # Tabela Sumário (results_*.csv)
    summary_df = load_run_summary(ds, run)
    summary_block = html.Div()
    fig_summary = None
    if not summary_df.empty:
        # Tenta criar um gráfico de barras para as acurácias do sumário
        try:
            acc_df = summary_df[summary_df["Metric"].str.contains("Accuracy")].copy()
            acc_df["Value_Num"] = acc_df["Value"].str.replace("%", "").astype(float)
            fig_summary = px.bar(acc_df, x="Metric", y="Value_Num", 
                                 title=f"Acurácias Finais — {run}",
                                 text_auto='.2f', color="Metric")
            fig_summary.update_layout(**LAYOUT_COMMON)
            fig_summary.update_yaxes(range=[0, 100])
        except:
            fig_summary = None

        summary_block = html.Div([
            section_title(f"📊 Sumário Final do Modelo — {run}"),
            html.Div([
                html.Div(dash_table.DataTable(
                    data=summary_df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in summary_df.columns],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#333", "color": "#fff",
                                  "fontWeight": "600", "fontFamily": FONT_SANS,
                                  "fontSize": "12px"},
                    style_cell={"fontFamily": FONT_SANS, "fontSize": "12px",
                                "padding": "8px", "textAlign": "left"},
                ), style={"flex": "1", "minWidth": "300px"}),
                html.Div(dcc.Graph(id="summary-graph", figure=fig_summary, config={"displayModeBar": False}) if fig_summary else html.Div(),
                         style={"flex": "1.5", "minWidth": "400px"})
            ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
        ], style={**card_style, "padding": "16px", "marginBottom": "16px"})

    # Comparação de Sumários (se selecionada)
    compare_summary_block = html.Div()
    if cmp_ds != "none" and cmp_run and cmp_run != "none":
        summary_df2 = load_run_summary(cmp_ds, cmp_run)
        if not summary_df2.empty and not summary_df.empty:
            # Tenta alinhar métricas para comparação
            m1 = summary_df.set_index("Metric")
            m2 = summary_df2.set_index("Metric")
            comp_sum = m1.join(m2, lsuffix=f" ({run})", rsuffix=f" ({cmp_run})", how="outer").reset_index()
            
            compare_summary_block = html.Div([
                section_title("⚖ Comparação de Sumários Finais"),
                dash_table.DataTable(
                    data=comp_sum.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in comp_sum.columns],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": HANDLE_COLOR, "color": "#fff",
                                  "fontWeight": "600", "fontFamily": FONT_SANS,
                                  "fontSize": "12px"},
                    style_cell={"fontFamily": FONT_SANS, "fontSize": "12px",
                                "padding": "8px", "textAlign": "left"},
                ),
            ], style={**card_style, "padding": "16px", "marginBottom": "16px"})

    charts = html.Div([
        two_col(fig_acc, fig_vac, id_prefix="base-acc"),
        two_col(fig_los, fig_vlo, id_prefix="base-los"),
        compare_block,
        summary_block,
        compare_summary_block,
        table_folds,
    ])

    return kpis, charts


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANÁLISE T-SNE
# ══════════════════════════════════════════════════════════════════════════════

# Dados sintéticos de T-SNE (substituir por load_tsne_data() real quando disponível)
_rng = np.random.default_rng(0)
_N   = 400
_lab = _rng.integers(0, 2, _N)
_X   = np.column_stack([
    _rng.normal(_lab * 4, 1.5, _N),
    _rng.normal(_lab * 3, 1.5, _N),
])
LABEL_MAP = {0: "Queratocisto", 1: "Dentígero"}
COLORS_CLS = ["rgb(18,97,228)", "rgb(252,124,76)"]


def build_tsne_fig(mode="full", highlight=None):
    traces = []
    for i, name in LABEL_MAP.items():
        idx  = np.where(_lab == i)[0]
        alpha = (
            _rng.uniform(0.2, 1.0, len(idx)).tolist()
            if mode == "conf" else None
        )
        traces.append(go.Scatter(
            x=_X[idx, 0], y=_X[idx, 1],
            mode="markers",
            marker=dict(
                color=COLORS_CLS[i],
                size=6,
                opacity=alpha if mode == "conf" else 0.75,
            ),
            name=name,
            customdata=idx,
        ))

    annotations = []
    if highlight is not None:
        annotations.append(dict(
            x=highlight[0], y=highlight[1],
            text="Upload", showarrow=True,
            arrowhead=2, ax=20, ay=-30,
            font=dict(size=13, color=PLUM),
            bgcolor="#fff", bordercolor=PLUM,
        ))

    fig = go.Figure(data=traces, layout=go.Layout(
        **{k: v for k, v in LAYOUT_COMMON.items()
           if k not in ("xaxis", "yaxis", "colorway")},
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        clickmode="event+select",
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        title="Visualização T-SNE — Queratocisto vs Dentígero",
        margin=dict(l=20, r=20, t=48, b=20),
    ))
    return fig


@app.callback(
    Output("tsne-graph", "figure"),
    Input("tsne-mode-sel", "value"),
    Input("tsne-upload",   "contents"),
)
def update_tsne_graph(mode, contents):
    highlight = None
    if contents:
        # Simula posição do upload no espaço T-SNE
        highlight = (_rng.normal(2, 1), _rng.normal(1.5, 1))
    return build_tsne_fig(mode, highlight)


@app.callback(
    Output("tsne-upload-info",  "children"),
    Output("tsne-pred-image",   "children"),
    Output("tsne-accuracy-area","children"),
    Output("tsne-pred-table",   "children"),
    Input("tsne-upload",        "contents"),
    State("tsne-upload",        "filename"),
)
def handle_tsne_upload(contents, fname):
    if not contents:
        placeholder = html.Div("Nenhuma imagem carregada.",
                               style={"color": MUTED, "fontFamily": FONT_SANS})
        return placeholder, placeholder, [], html.Div()

    # Info do arquivo
    info = html.Div([
        html.Span(f"📄 {fname}", style={"fontSize": "12px",
                                        "fontFamily": FONT_MONO,
                                        "color": HANDLE_COLOR}),
    ])

    # Exibe imagem carregada
    img_el = html.Img(src=contents, style={
        "maxWidth": "100%", "maxHeight": "200px",
        "borderRadius": "6px", "border": "1px solid #E8E8E8",
    })

    # Simula predição (substituir por model.predict real)
    pred_cls  = np.random.choice(list(LABEL_MAP.values()))
    prob      = np.random.uniform(0.70, 0.99)
    real_cls  = np.random.choice(list(LABEL_MAP.values()))
    correct   = pred_cls == real_cls
    cor_color = "rgb(34,139,34)" if correct else "rgb(200,40,40)"

    accuracy_kpis = [
        kpi_card("Predição",   pred_cls),
        kpi_card("Confiança",  f"{prob*100:.1f}%"),
        kpi_card("Real",       real_cls),
        kpi_card("Resultado",  "✓ Correto" if correct else "✗ Incorreto"),
    ]

    # Tabela de predições simulada
    rows = [{"Classe": k, "Probabilidade": f"{v:.4f}"}
            for k, v in zip(LABEL_MAP.values(),
                            np.random.dirichlet([1, 1]))]
    pred_table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in ["Classe", "Probabilidade"]],
        style_header={"backgroundColor": PLUM, "color": "#fff",
                      "fontWeight": "600", "fontFamily": FONT_SANS,
                      "fontSize": "12px"},
        style_cell={"fontFamily": FONT_SANS, "fontSize": "13px",
                    "padding": "10px", "textAlign": "left"},
        style_data_conditional=[
            {"if": {"filter_query": f'{{Classe}} = "{pred_cls}"'},
             "backgroundColor": "#fdf5f9", "fontWeight": "600",
             "color": cor_color}
        ],
    )

    return info, img_el, accuracy_kpis, pred_table


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

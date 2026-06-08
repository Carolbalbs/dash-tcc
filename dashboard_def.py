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
# Usando paths relativos ao diretório do script para maior flexibilidade
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIM_PATH = os.path.join(BASE_DIR, "data/dimensoes-dataset")
RESULTS_TRAIN_BASE = os.path.join(BASE_DIR, "data/results-train")

# Mapeamento para as subpastas físicas
SESSION_FOLDER_MAP = {
    "Novo": "dataset-novo",
    "Antigo": "dataset-antigo",
    "3 Classes": "dataset-3class"
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

def get_training_runs(dataset_type, optimizer):
    subfolder = SESSION_FOLDER_MAP.get(dataset_type)
    if not subfolder:
        return []
    
    target_path = os.path.join(RESULTS_TRAIN_BASE, subfolder)
    if not os.path.exists(target_path):
        return []
    
    all_runs = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
    
    # Filtro de otimizador
    runs = [r for r in all_runs if optimizer.upper() in r.upper()]
    
    return sorted(runs)


def get_run_options(dataset_type, optimizer):
    runs = get_training_runs(dataset_type, optimizer)
    options = []
    for r in runs:
        parts = r.split("_")
        # Se for results_ADAM_InceptionV3_DATE
        date_label = "_".join([p for p in parts[-3:] if p]) if len(parts) >= 3 else r
        options.append({"label": date_label, "value": r})
    return options


def load_training_data(dataset_type, run_name):
    if not run_name or not dataset_type:
        return pd.DataFrame()
    
    subfolder = SESSION_FOLDER_MAP.get(dataset_type)
    path = os.path.join(RESULTS_TRAIN_BASE, subfolder, run_name)
    
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


def load_run_summary(dataset_type, run_name):
    """Carrega o arquivo results_*.csv se existir na pasta do treino."""
    if not run_name or not dataset_type:
        return pd.DataFrame()
    
    subfolder = SESSION_FOLDER_MAP.get(dataset_type)
    path = os.path.join(RESULTS_TRAIN_BASE, subfolder, run_name)
    
    files = glob.glob(os.path.join(path, "results_*.csv"))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[0], sep=";")
        return df
    except Exception as e:
        print(f"Erro ao carregar sumário {files[0]}: {e}")
        return pd.DataFrame()


def load_dim_csv(dataset_type):
    path = os.path.join(DATASET_DIM_PATH, f"image_dimensions_dataset_{dataset_type}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


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


def get_training_session(session_id, title):
    return html.Div([
        section_title(f"📁 {title}"),
        # Filtros (Adam e RMSProp)
        html.Div([
            html.Div([
                dropdown(f"{session_id}-adam-sel", [], None, "Otimizador ADAM (por data)"),
            ], style={"flex": "1", "minWidth": "240px"}),
            html.Div([
                dropdown(f"{session_id}-rmsprop-sel", [], None, "Otimizador RMSPROP (por data)"),
            ], style={"flex": "1", "minWidth": "240px"}),
        ], style={**card_style, "padding": "16px", "display": "flex",
                  "flexWrap": "wrap", "gap": "16px", "marginBottom": "16px"}),

        # Gráficos
        html.Div(id=f"{session_id}-charts"),
    ], style={"marginBottom": "48px"})


def get_training_summary_metrics(dataset_type, run_name):
    """Extrai métricas específicas do arquivo results_*.csv para os KPI Cards."""
    if not run_name or not dataset_type:
        return {}
    
    subfolder = SESSION_FOLDER_MAP.get(dataset_type)
    path = os.path.join(RESULTS_TRAIN_BASE, subfolder, run_name)
    files = glob.glob(os.path.join(path, "results_*.csv"))
    if not files:
        return {}
    
    metrics = {}
    try:
        # Lendo como texto para lidar com o formato misto do CSV
        with open(files[0], 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(';')
                if len(parts) < 2: continue
                key = parts[0].strip()
                val = parts[1].strip()
                
                if "Média Accuracy" in key:
                    metrics["avg_acc"] = val
                elif "Desvio Padrão" in key:
                    metrics["std_dev"] = val
                elif "Tempo de execução" in key:
                    metrics["exec_time"] = f"{val} min"
                elif key.startswith("Fold"):
                    # Encontra a melhor acurácia entre os folds
                    try:
                        acc_val = float(val.replace('%', ''))
                        if "best_fold" not in metrics or acc_val > float(metrics["best_fold"].replace('%', '')):
                            metrics["best_fold"] = val
                    except: pass
    except Exception as e:
        print(f"Erro ao extrair KPIs de {run_name}: {e}")
    return metrics


def build_comparison_charts(dataset_type, adam_run, rmsprop_run):
    df_adam = load_training_data(dataset_type, adam_run)
    df_rms = load_training_data(dataset_type, rmsprop_run)
    
    if df_adam.empty and df_rms.empty:
        return html.Div("Nenhum dado encontrado para esta sessão. Verifique os dropdowns acima.",
                        style={"color": MUTED, "fontFamily": FONT_SANS, "padding": "20px",
                               "textAlign": "center", **card_style})

    # Extração de KPIs
    kpi_data_adam = get_training_summary_metrics(dataset_type, adam_run)
    kpi_data_rms = get_training_summary_metrics(dataset_type, rmsprop_run)

    def make_kpi_group(data, opt_name, color):
        if not data: return html.Div()
        return html.Div([
            html.Div(f"🚀 Métricas {opt_name}", style={
                "fontSize": "14px", "fontWeight": "700", "color": color,
                "marginBottom": "10px", "fontFamily": FONT_SANS, "marginLeft": "4px"
            }),
            html.Div([
                kpi_card("Melhor Fold (%)", data.get("best_fold", "—")),
                kpi_card("Média Accuracy",   data.get("avg_acc", "—")),
                kpi_card("Desvio Padrão",    data.get("std_dev", "—")),
                kpi_card("Tempo Total",      data.get("exec_time", "—")),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}),
        ], style={"flex": "1", "minWidth": "300px", "marginBottom": "20px"})

    kpi_row = html.Div([
        make_kpi_group(kpi_data_adam, "ADAM", PLUM),
        make_kpi_group(kpi_data_rms, "RMSPROP", "#12a1e4"),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "10px"})

    # Função interna para gerar gráficos pareados
    def make_pair(metric, title, id_prefix):
        fig_adam = px.line(df_adam, x="epoch", y=metric, color="Fold", title=f"ADAM - {title}") if not df_adam.empty else go.Figure()
        fig_rms = px.line(df_rms, x="epoch", y=metric, color="Fold", title=f"RMSPROP - {title}") if not df_rms.empty else go.Figure()
        
        for f in [fig_adam, fig_rms]:
            f.update_layout(**LAYOUT_COMMON)
            
        # Adiciona o dataset_type ao prefixo para evitar IDs duplicados entre sessões
        clean_ds = dataset_type.lower().replace(" ", "-")
        return two_col(fig_adam, fig_rms, height=300, id_prefix=f"{clean_ds}-{id_prefix}")

    # Tabelas de sumário
 
    
    summary_row = html.Div()
    
    return html.Div([
        kpi_row,
        summary_row,
        make_pair("accuracy", "Acurácia de Treino", "acc-tr"),
        make_pair("val_accuracy", "Acurácia de Validação", "acc-val"),
        make_pair("loss", "Loss de Treino", "loss-tr"),
        make_pair("val_loss", "Loss de Validação", "loss-val"),
    ])


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
            get_training_session("ds-novo", "Dataset Novo"),
            get_training_session("ds-antigo", "Dataset Antigo"),
            get_training_session("ds-3cl", "Dataset com 3 Classes"),
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
# TAB 2 — ANÁLISE DE TREINAMENTO (SESSÕES)
# ══════════════════════════════════════════════════════════════════════════════

# --- Session: NOVO ---
@app.callback(
    Output("ds-novo-adam-sel", "options"),
    Output("ds-novo-adam-sel", "value"),
    Input("main-tabs", "value"),
)
def update_novo_adam(tab):
    if tab != "tab-training": return [], None
    opts = get_run_options("Novo", "ADAM")
    return opts, (opts[0]["value"] if opts else None)

@app.callback(
    Output("ds-novo-rmsprop-sel", "options"),
    Output("ds-novo-rmsprop-sel", "value"),
    Input("main-tabs", "value"),
)
def update_novo_rms(tab):
    if tab != "tab-training": return [], None
    opts = get_run_options("Novo", "RMSPROP")
    return opts, (opts[0]["value"] if opts else None)

@app.callback(
    Output("ds-novo-charts", "children"),
    Input("ds-novo-adam-sel", "value"),
    Input("ds-novo-rmsprop-sel", "value"),
)
def update_novo_charts(adam, rms):
    return build_comparison_charts("Novo", adam, rms)


# --- Session: ANTIGO ---
@app.callback(
    Output("ds-antigo-adam-sel", "options"),
    Output("ds-antigo-adam-sel", "value"),
    Input("main-tabs", "value"),
)
def update_antigo_adam(tab):
    if tab != "tab-training": return [], None
    opts = get_run_options("Antigo", "ADAM")
    return opts, (opts[0]["value"] if opts else None)

@app.callback(
    Output("ds-antigo-rmsprop-sel", "options"),
    Output("ds-antigo-rmsprop-sel", "value"),
    Input("main-tabs", "value"),
)
def update_antigo_rms(tab):
    if tab != "tab-training": return [], None
    opts = get_run_options("Antigo", "RMSPROP")
    return opts, (opts[0]["value"] if opts else None)

@app.callback(
    Output("ds-antigo-charts", "children"),
    Input("ds-antigo-adam-sel", "value"),
    Input("ds-antigo-rmsprop-sel", "value"),
)
def update_antigo_charts(adam, rms):
    return build_comparison_charts("Antigo", adam, rms)


# --- Session: 3 CLASSES ---
@app.callback(
    Output("ds-3cl-adam-sel", "options"),
    Output("ds-3cl-adam-sel", "value"),
    Input("main-tabs", "value"),
)
def update_3cl_adam(tab):
    if tab != "tab-training": return [], None
    opts = get_run_options("3 Classes", "ADAM")
    return opts, (opts[0]["value"] if opts else None)

@app.callback(
    Output("ds-3cl-rmsprop-sel", "options"),
    Output("ds-3cl-rmsprop-sel", "value"),
    Input("main-tabs", "value"),
)
def update_3cl_rms(tab):
    if tab != "tab-training": return [], None
    opts = get_run_options("3 Classes", "RMSPROP")
    return opts, (opts[0]["value"] if opts else None)

@app.callback(
    Output("ds-3cl-charts", "children"),
    Input("ds-3cl-adam-sel", "value"),
    Input("ds-3cl-rmsprop-sel", "value"),
)
def update_3cl_charts(adam, rms):
    return build_comparison_charts("3 Classes", adam, rms)



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

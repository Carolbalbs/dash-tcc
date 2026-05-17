# Dashboard de Análise TCC — Cistos Odontogênicos

Dashboard interativo em Dash/Plotly para análise de datasets, treinamentos e exploração T-SNE na classificação de cistos odontogênicos (Queratocisto vs Dentígero).

---

## Instalação

```bash
pip install dash plotly pandas numpy
python dashboard_unificado.py
# Acesse: http://localhost:8050
```

---

## Estrutura esperada de pastas

```
projeto/
│
├── dashboard_unificado.py          ← script principal
│
├── dimensoes-dataset/              ← CSVs de dimensões das imagens
│   ├── image_dimensions_dataset_antigo.csv
│   └── image_dimensions_dataset_novo.csv
│
├── Resultado_dataset-antigo/       ← resultados dos treinamentos no dataset antigo
│   ├── Treinamento_ADAM_InceptionV3_.../
│   │   ├── log_Fold1.csv
│   │   ├── log_Fold2.csv
│   │   └── ...
│   └── outro_treinamento.../
│
└── Resultado_dataset-novo/         ← resultados dos treinamentos no dataset novo
    ├── Treinamento_ADAM_.../
    │   ├── log_Fold1.csv
    │   └── ...
    └── ...
```

---

## Como mudar os paths

Todos os caminhos ficam nas **três constantes no topo** do arquivo `dashboard_unificado.py`, logo abaixo dos imports:

```python
# ── Configurações de paths ──────────────────────────────────────────
DATASET_DIM_PATH = "dimensoes-dataset"
RESULTS_PATHS = {
    "Antigo": "Resultado_dataset-antigo",
    "Novo":   "Resultado_dataset-novo",
}
```

### `DATASET_DIM_PATH`

Pasta onde ficam os CSVs com as dimensões das imagens (Tab 1).

O dashboard vai procurar arquivos com o padrão:

```
{DATASET_DIM_PATH}/image_dimensions_dataset_{antigo|novo}.csv
```

**Exemplos de como alterar:**

```python
# Caminho relativo numa subpasta diferente
DATASET_DIM_PATH = "data/dimensoes"

# Caminho absoluto
DATASET_DIM_PATH = "/home/usuario/tcc/dados/dimensoes-dataset"

# Windows
DATASET_DIM_PATH = r"C:\Users\usuario\TCC\dimensoes-dataset"
```

---

### `RESULTS_PATHS`

Dicionário que mapeia o nome exibido no dropdown para a pasta de resultados de treinamento (Tab 2).

Cada subpasta dentro do caminho é detectada automaticamente como um **treinamento** no dropdown.

**Exemplos de como alterar:**

```python
# Mudar só um dos datasets
RESULTS_PATHS = {
    "Antigo": "resultados/dataset-v1",
    "Novo":   "resultados/dataset-v2",
}

# Caminho absoluto
RESULTS_PATHS = {
    "Antigo": "/home/usuario/tcc/Resultado_dataset-antigo",
    "Novo":   "/home/usuario/tcc/Resultado_dataset-novo",
}

# Adicionar um terceiro dataset
RESULTS_PATHS = {
    "Antigo":    "Resultado_dataset-antigo",
    "Novo":      "Resultado_dataset-novo",
    "Aumentado": "Resultado_dataset-aumentado",   # ← nova entrada
}
```

> Ao adicionar uma nova entrada no dicionário, ela aparece automaticamente nos dropdowns de Dataset nas Tabs 1 e 2 — não é necessário alterar mais nada.

---

## Formato esperado dos CSVs

### CSVs de dimensões (`DATASET_DIM_PATH`)

| Coluna      | Tipo   | Descrição                        |
|-------------|--------|----------------------------------|
| `filename`  | string | Nome do arquivo de imagem        |
| `width`     | int    | Largura em pixels                |
| `height`    | int    | Altura em pixels                 |
| `directory` | string | Classe/pasta da imagem           |

### CSVs de log de treinamento (`log_FoldN.csv`)

Separador: `;` (ponto e vírgula)

| Coluna         | Tipo  | Descrição                        |
|----------------|-------|----------------------------------|
| `epoch`        | int   | Número da época                  |
| `accuracy`     | float | Acurácia de treino               |
| `val_accuracy` | float | Acurácia de validação            |
| `loss`         | float | Loss de treino                   |
| `val_loss`     | float | Loss de validação                |

> Se o seu CSV usar vírgula como separador, mude `sep=";"` para `sep=","` na função `load_training_data()`.

---

## Conectando o modelo real na Tab T-SNE

A Tab 3 (Análise T-SNE) usa dados sintéticos por padrão. Para conectar ao modelo real do `odontogenic_explorer.py`, localize o bloco comentado abaixo de `# Dados sintéticos de T-SNE` e substitua:

```python
# ANTES — dados sintéticos
_rng = np.random.default_rng(0)
_N   = 400
_lab = _rng.integers(0, 2, _N)
_X   = np.column_stack([...])

# DEPOIS — dados reais
from odontogenic_helpers import load_tsne_data
_X, _lab, image_paths, linear_model = load_tsne_data()
```

E no callback `handle_tsne_upload`, substitua a simulação de predição pelo modelo real:

```python
# ANTES — simulação
pred_cls = np.random.choice(list(LABEL_MAP.values()))
prob     = np.random.uniform(0.70, 0.99)

# DEPOIS — modelo real
from tensorflow.keras.models import load_model
model = load_model("caminho/para/best_model.keras")
# ... pré-processar imagem e chamar model.predict()
```

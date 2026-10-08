from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "modelo_final.joblib"
METADATA_PATH = ROOT / "models" / "model_info.json"
DATA_PATH = ROOT / "data" / "winequality-red.csv"

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

LABELS = {
    "fixed acidity": "Fixed acidity",
    "volatile acidity": "Volatile acidity",
    "citric acid": "Citric acid",
    "residual sugar": "Residual sugar",
    "chlorides": "Chlorides",
    "free sulfur dioxide": "Free sulfur dioxide",
    "total sulfur dioxide": "Total sulfur dioxide",
    "density": "Density",
    "pH": "pH",
    "sulphates": "Sulphates",
    "alcohol": "Alcohol (% vol.)",
}

DEFAULTS = {
    "fixed acidity": 8.32,
    "volatile acidity": 0.53,
    "citric acid": 0.27,
    "residual sugar": 2.54,
    "chlorides": 0.087,
    "free sulfur dioxide": 15.87,
    "total sulfur dioxide": 46.47,
    "density": 0.9968,
    "pH": 3.31,
    "sulphates": 0.66,
    "alcohol": 10.42,
}

FALLBACK_LIMITS = {
    "fixed acidity": (4.0, 16.0, 0.1, "%.2f"),
    "volatile acidity": (0.1, 2.0, 0.01, "%.2f"),
    "citric acid": (0.0, 1.5, 0.01, "%.2f"),
    "residual sugar": (0.5, 16.0, 0.1, "%.2f"),
    "chlorides": (0.01, 0.7, 0.001, "%.3f"),
    "free sulfur dioxide": (1.0, 80.0, 1.0, "%.2f"),
    "total sulfur dioxide": (5.0, 300.0, 1.0, "%.2f"),
    "density": (0.9900, 1.0100, 0.0001, "%.4f"),
    "pH": (2.7, 4.2, 0.01, "%.2f"),
    "sulphates": (0.2, 2.2, 0.01, "%.2f"),
    "alcohol": (8.0, 15.0, 0.1, "%.2f"),
}


def friendly_model_name(model):
    name = model.__class__.__name__
    mapping = {
        "RandomForestRegressor": "Random Forest",
        "XGBRegressor": "XGBoost",
        "LGBMRegressor": "LightGBM",
    }
    return mapping.get(name, name)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data
def load_limits():
    """Usa os limites observados na base quando ela estiver disponível."""
    if not DATA_PATH.exists():
        return FALLBACK_LIMITS

    try:
        df = pd.read_csv(DATA_PATH, sep=";")
        limits = {}
        for feature in FEATURES:
            lo = float(df[feature].min())
            hi = float(df[feature].max())
            step = FALLBACK_LIMITS[feature][2]
            fmt = FALLBACK_LIMITS[feature][3]
            limits[feature] = (lo, hi, step, fmt)
        return limits
    except Exception:
        return FALLBACK_LIMITS


# -----------------------------------------------------------------------------
# Cabeçalho e carregamento
# -----------------------------------------------------------------------------
st.title("🍷 Wine Quality Predictor")
st.write(
    "Previsão da qualidade de **vinhos tintos Vinho Verde** com base em suas "
    "características físico-químicas. A aplicação utiliza o mesmo artefato final "
    "avaliado no notebook do Checkpoint 5."
)

if not MODEL_PATH.exists():
    st.error(
        "O arquivo `models/modelo_final.joblib` ainda não existe. Execute o notebook "
        "até o Exercício 7 para gerar o modelo final e suas informações de avaliação."
    )
    st.stop()

modelo = load_model()
metadata = load_metadata()
limits = load_limits()

model_name = metadata.get("modelo", friendly_model_name(modelo))
configuration = metadata.get("configuracao", "Modelo final salvo pelo notebook")

# Resumo do modelo
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Modelo final", model_name)
with m2:
    st.metric("Configuração", configuration)
with m3:
    mae = metadata.get("mae_teste")
    st.metric("MAE no teste", f"{mae:.3f}" if isinstance(mae, (int, float)) else "—")
with m4:
    r2 = metadata.get("r2_teste")
    st.metric("R² no teste", f"{r2:.3f}" if isinstance(r2, (int, float)) else "—")

if not metadata:
    st.caption(
        "As métricas e o caso de paridade aparecerão automaticamente após executar a "
        "versão atualizada do notebook até o final."
    )

st.divider()

# -----------------------------------------------------------------------------
# Caso de paridade
# -----------------------------------------------------------------------------
parity = metadata.get("paridade", {}) if metadata else {}

if parity and parity.get("entrada"):
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("🧪 Carregar caso de paridade", use_container_width=True):
            for feature, value in parity["entrada"].items():
                st.session_state[f"input_{feature}"] = float(value)
            st.session_state["parity_loaded"] = True
            st.rerun()
    with col_b:
        st.caption(
            "Carrega uma observação real do conjunto de teste salva pelo notebook. "
            "Ela pode ser usada na apresentação para demonstrar a paridade notebook–Streamlit."
        )

# Inicializa valores dos campos
for feature in FEATURES:
    key = f"input_{feature}"
    if key not in st.session_state:
        st.session_state[key] = float(DEFAULTS[feature])

# -----------------------------------------------------------------------------
# Formulário de entrada
# -----------------------------------------------------------------------------
st.subheader("Características físico-químicas")

columns = st.columns(3)
feature_groups = [FEATURES[:4], FEATURES[4:8], FEATURES[8:]]

for col, group in zip(columns, feature_groups):
    with col:
        for feature in group:
            lo, hi, step, fmt = limits[feature]
            # Garante que o valor carregado de paridade esteja dentro dos limites.
            current = float(st.session_state[f"input_{feature}"])
            lo = min(float(lo), current)
            hi = max(float(hi), current)
            st.number_input(
                LABELS[feature],
                min_value=float(lo),
                max_value=float(hi),
                step=float(step),
                format=fmt,
                key=f"input_{feature}",
            )

entrada = pd.DataFrame([
    {feature: float(st.session_state[f"input_{feature}"]) for feature in FEATURES}
])

if st.button("Prever qualidade", type="primary", use_container_width=True):
    pred = float(modelo.predict(entrada)[0])

    st.subheader("Resultado")
    st.metric("Qualidade prevista", f"{pred:.2f} / 10")

    if pred < 5:
        st.info("O modelo estima uma qualidade abaixo da faixa mais comum da base.")
    elif pred < 7:
        st.success("O modelo estima uma qualidade intermediária.")
    else:
        st.success("O modelo estima uma qualidade elevada para o padrão da base.")

    # Verificação explícita da paridade quando o exemplo do notebook estiver carregado.
    if parity and st.session_state.get("parity_loaded"):
        expected = parity.get("previsao_notebook")
        actual_y = parity.get("quality_real")
        if isinstance(expected, (int, float)):
            same = bool(np.isclose(pred, float(expected), rtol=1e-7, atol=1e-8))
            if same:
                st.success(
                    f"✅ Paridade confirmada: Streamlit = {pred:.6f} e notebook = {float(expected):.6f}."
                )
            else:
                st.warning(
                    f"⚠️ Divergência de paridade: Streamlit = {pred:.6f} e notebook = {float(expected):.6f}."
                )
        if isinstance(actual_y, (int, float)):
            st.caption(f"Valor real dessa observação no conjunto de teste: **{actual_y}**")

    with st.expander("Entradas utilizadas", expanded=False):
        display_input = entrada.T.reset_index()
        display_input.columns = ["Variável", "Valor"]
        st.dataframe(display_input, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Informações acadêmicas e técnicas
# -----------------------------------------------------------------------------
with st.expander("ℹ️ Informações do modelo"):
    st.markdown(
        f"""
**Problema:** regressão da variável `quality`.

**Base:** Wine Quality — Red Wine (UCI Machine Learning Repository).

**Atributos:** 11 características físico-químicas.

**Métrica principal:** MAE (Mean Absolute Error).

**Modelo selecionado:** {model_name}.

**Estratégia/configuração:** {configuration}.
"""
    )

    rmse = metadata.get("rmse_teste")
    mae_cv = metadata.get("mae_cv")
    std_cv = metadata.get("std_cv")

    if any(isinstance(v, (int, float)) for v in [mae, rmse, r2, mae_cv, std_cv]):
        st.markdown("**Resultados registrados pelo notebook:**")
        metrics_df = pd.DataFrame(
            {
                "Métrica": ["MAE teste", "RMSE teste", "R² teste", "MAE CV", "Desvio-padrão CV"],
                "Valor": [
                    f"{mae:.4f}" if isinstance(mae, (int, float)) else "—",
                    f"{rmse:.4f}" if isinstance(rmse, (int, float)) else "—",
                    f"{r2:.4f}" if isinstance(r2, (int, float)) else "—",
                    f"{mae_cv:.4f}" if isinstance(mae_cv, (int, float)) else "—",
                    f"{std_cv:.4f}" if isinstance(std_cv, (int, float)) else "—",
                ],
            }
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with st.expander("📌 Sobre a previsão"):
    st.write(
        "A saída representa a estimativa da nota de qualidade do vinho a partir das "
        "variáveis informadas. A aplicação tem finalidade exclusivamente acadêmica e "
        "demonstra a disponibilização do modelo desenvolvido no Checkpoint 5."
    )

st.caption("Projeto acadêmico — Data Science & Statistical Computing — FIAP 2026")

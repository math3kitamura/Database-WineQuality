"""Treinamento reproduzível do modelo final.

Execute o notebook primeiro para realizar toda a análise acadêmica. Este script é uma
alternativa para gerar um artefato de produção de forma direta usando uma configuração
robusta de Random Forest caso o arquivo models/modelo_final.joblib ainda não exista.
"""
from pathlib import Path
import urllib.request
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "winequality-red.csv"
MODEL = ROOT / "models" / "modelo_final.joblib"

if not DATA.exists():
    DATA.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        DATA,
    )

df = pd.read_csv(DATA, sep=";").drop_duplicates().reset_index(drop=True)
X = df.drop(columns="quality")
y = df["quality"]

modelo = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
modelo.fit(X, y)
MODEL.parent.mkdir(exist_ok=True)
joblib.dump(modelo, MODEL)
print(f"Modelo salvo em {MODEL}")

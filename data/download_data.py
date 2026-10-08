from pathlib import Path
import urllib.request

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DESTINO = Path(__file__).resolve().parent / "winequality-red.csv"

if DESTINO.exists():
    print(f"Arquivo já existe: {DESTINO}")
else:
    print("Baixando Wine Quality (red wine) da UCI...")
    urllib.request.urlretrieve(URL, DESTINO)
    print(f"Base salva em: {DESTINO}")

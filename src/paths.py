from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent

DEFAULT_SERIES = ROOT_DIR / "datasets" / "MixedBag" / "GreatBarbet1F_Jog_1800_200.txt"
DEFAULT_RESULTS = ROOT_DIR / "results"
DEFAULT_MODEL = PROJECT_DIR / "models" / "salto.pt"

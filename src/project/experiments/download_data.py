from pathlib import Path
import gdown

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

ELCENTRO_ID = "112GZdUJU1iRGKiu0Qq-TLS6HAru3EApt"
KOBE_ID = "1pPEKXcbzh9706xYm1AfdGQ_1wspSuNa0"


def _download_if_missing(file_id: str, output_path: Path) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {output_path.name} to: {output_path}")
        gdown.download(url, str(output_path), quiet=False)
    else:
        print(f"Using existing file: {output_path}")

    return output_path


def ensure_elcentro() -> Path:
    return _download_if_missing(ELCENTRO_ID, DATA_DIR / "elcentro.dat")


def ensure_kobe() -> Path:
    return _download_if_missing(KOBE_ID, DATA_DIR / "kobe.at2")


def ensure_all_earthquakes():
    elcentro_path = ensure_elcentro()
    kobe_path = ensure_kobe()
    return elcentro_path, kobe_path


if __name__ == "__main__":
    print(f"Repo root: {REPO_ROOT}")
    print(f"Data dir:   {DATA_DIR}")
    ensure_all_earthquakes()
    print("Done.")

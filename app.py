from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "SRC"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Streamlit_App import main


if __name__ == "__main__":
    main()
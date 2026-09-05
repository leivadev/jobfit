import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

load_dotenv(Path(__file__).parent.parent / ".env")

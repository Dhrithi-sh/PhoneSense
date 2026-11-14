import os
import sys
from pathlib import Path

# Make the dhvani package importable when running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The CLI and app use the real index when it's built; tests stay on the
# 20-article sample so they give the same answers on every machine.
os.environ["DHVANI_INDEX"] = "sample"

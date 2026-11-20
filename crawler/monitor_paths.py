"""Default artifact locations for the deployed news monitor."""
from pathlib import Path


OUTPUT_ROOT = Path("/opt/riseai/data/crawler")
FEATURES_DIR = OUTPUT_ROOT / "features"
VALIDATION_DIR = OUTPUT_ROOT / "validation_monitor"
LOG_DIR = OUTPUT_ROOT / "log"

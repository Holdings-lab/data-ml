import subprocess
import sys
from pathlib import Path


def main() -> None:
    scraper_path = Path(__file__).with_name("scraper.py")
    subprocess.run([sys.executable, str(scraper_path)], check=True)


if __name__ == "__main__":
    main()

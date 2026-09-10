"""The checked-in CSV files are already usable.

This placeholder exists so the project has a clear extension point for larger
synthetic demos or for converting competition data into the unified schema.
"""

from pathlib import Path


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    print(f"Demo data is ready in {data_dir}")


if __name__ == "__main__":
    main()

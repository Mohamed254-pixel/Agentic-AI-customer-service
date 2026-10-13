import kagglehub
from pathlib import Path

# Download Instacart dataset into a new folder
path = kagglehub.dataset_download(
    "psparks/instacart-market-basket-analysis",
    output_dir="data/raw/instacart"
)

print("Dataset downloaded to:", path)

# Check downloaded CSV files
csv_files = list(Path(path).rglob("*.csv"))

for file in csv_files:
    print(f"{file.name}: {file.stat().st_size / 1024**2:.2f} MB")

print(f"\nTotal CSV files: {len(csv_files)}")
import os
import re
from pathlib import Path

# Rename CSV files: trace-00<number>.csv -> trace-001-00<number>.csv
csv_pattern = re.compile(r'^trace-00(\d+)\.csv$')
csv_files = Path("../../TE-Experiments").glob("**/trace-00*.csv")

for csv_file in csv_files:
    match = csv_pattern.match(csv_file.name)
    if match:
        number = match.group(1)
        new_name = f"trace-001-00{number}.csv"
        new_path = csv_file.parent / new_name
        print(f"Renaming: {csv_file.name} -> {new_name}")
        csv_file.rename(new_path)

# Rename .gz files: state-00<number>-<7-numbers>.gz -> state-001-00<number>-<7-numbers>.gz
gz_pattern = re.compile(r'^state-00(\d+)-(\d{7})\.gz$')
gz_files = Path("../../TE-Experiments").glob("**/state-00*-*.gz")

for gz_file in gz_files:
    match = gz_pattern.match(gz_file.name)
    if match:
        number = match.group(1)
        seven_nums = match.group(2)
        new_name = f"state-001-00{number}-{seven_nums}.gz"
        new_path = gz_file.parent / new_name
        print(f"Renaming: {gz_file.name} -> {new_name}")
        gz_file.rename(new_path)

print("Done!")
"""
Creates a LARGE test CSV (default 20,000 rows) in the same column layout as the
USGS download, so you can test speed before/without downloading real data.

!! The data is RANDOMLY GENERATED (not real earthquakes). Use the real USGS
!! file for your actual submission - see README.md.

Usage:  python generate_synthetic_data.py [rows] [output.csv]
"""
import csv
import random
import sys
from datetime import datetime, timedelta

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
output = sys.argv[2] if len(sys.argv) > 2 else "synthetic_earthquakes.csv"

random.seed(2024)
start = datetime(2015, 1, 1)
span_seconds = 10 * 365 * 24 * 3600

with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["time", "latitude", "longitude", "depth", "mag", "magType", "place", "id"])
    for number in range(rows):
        when = start + timedelta(seconds=random.randint(0, span_seconds))
        magnitude = min(9.5, round(2.5 + random.expovariate(1.6), 1))   # many small, few big
        writer.writerow([when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                         round(random.uniform(-60, 70), 3), round(random.uniform(-180, 180), 3),
                         round(random.uniform(0, 300), 1), magnitude, "mb",
                         "synthetic location", "syn%06d" % number])
print("Wrote %d rows to %s" % (rows, output))

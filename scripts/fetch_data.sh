#!/usr/bin/env bash
set -e

# Get Data Snapshot
DATE=${1:-2025/03/08}
BASE_URL="https://ton.twimg.com/birdwatch-public-data/${DATE}/notes"


mkdir -p data/raw

# Download single snapshot (no sharding)
URL="${BASE_URL}/notes-00000.zip"
wget -c "${URL}" -O data/raw/notes-00000.zip
unzip -n data/raw/notes-00000.zip -d data/raw/
#! /bin/bash

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

RESULT_JSON="${1:-./result/result_luna16_fold0.json}"
DATASET_JSON="${2:-./LUNA16_datasplit/dataset_fold0.json}"
ANNOTATIONS_CSV="${3:-./evaluation_luna16/annotations/annotations.csv}"
ANNOTATIONS_EXCLUDED_CSV="${4:-./evaluation_luna16/annotations/annotations_excluded.csv}"

RESULT_CSV="./result/result_luna16_fold0.csv"
SERIESUIDS_CSV="./result/seriesuids_fold0.csv"
OUTPUT_DIR="./result/eval_luna16_fold0_scores"

python ./luna16_post_combine_cross_fold_results.py \
  -i "${RESULT_JSON}" \
  -o "${RESULT_CSV}"

python - "${DATASET_JSON}" "${SERIESUIDS_CSV}" <<'PY'
import json
import os
import sys

dataset_json = sys.argv[1]
seriesuids_csv = sys.argv[2]

with open(dataset_json, "r") as f:
    data = json.load(f)

seriesuids = []
seen = set()
for item in data.get("validation", []):
    image = item["image"]
    seriesuid = os.path.basename(image)
    if seriesuid.endswith(".nii.gz"):
        seriesuid = seriesuid[:-7]
    if seriesuid not in seen:
        seen.add(seriesuid)
        seriesuids.append(seriesuid)

with open(seriesuids_csv, "w") as f:
    for seriesuid in seriesuids:
        f.write(seriesuid + "\n")
PY

mkdir -p "${OUTPUT_DIR}"
python ./evaluation_luna16/noduleCADEvaluationLUNA16.py \
  "${ANNOTATIONS_CSV}" \
  "${ANNOTATIONS_EXCLUDED_CSV}" \
  "${SERIESUIDS_CSV}" \
  "${RESULT_CSV}" \
  "${OUTPUT_DIR}"

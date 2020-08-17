import logging
import glob
from convert import check_gt_and_generate_copynet_file, STATUS
import os
from os.path import join
import pandas as pd
import sys
from joblib import Parallel, delayed

# Read collection dir and csv
# Read split files
# check if tmp files already exist
# Create copynet files
# Run allennlp with provided config
# Predict on test set
# Convert test set to jsons
# Run mleval on predicted jsons

logger = logging.getLogger("Logger")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO, stream=sys.stdout
)
logger.setLevel(logging.INFO)

INPUT_DIR = "/home/cehmann/workspaces/copynet/amtrust/input"
OUTPUT_DIR = "/home/cehmann/workspaces/copynet/amtrust/output"
INTERMEDIATE = "intermediate"
COLLECTION_ID = "collection"
SPLITS = {"train": "train_uuids.txt", "val": "val_uuids.txt", "test": "test_uuids.txt"}
CSV = "AnswerKey-7.7.20-corrected.csv"
OCR_NAME = "ocr-page.xml"
DEFAULT_FIELDS = {"ClaimNumber",
"LossDate",
"LossStateCd",
"Policy",
"RegulatoryClaimNumber",
"ConvertedClaimNumber",
"VIN",
"InsuredName",
"CompanyName",
"FirstName",
"LastName",
"DOB",
"SSN",
"Address1"}

df = pd.read_csv(join(INPUT_DIR, CSV), dtype=str)

XMLS_PER_SPLIT = []
for split, file in SPLITS.items():
    with open(join(INPUT_DIR, file)) as f:
        uuids = f.read().splitlines()
        results = Parallel(n_jobs=1)(
            delayed(check_gt_and_generate_copynet_file)(
                join(OUTPUT_DIR, INTERMEDIATE, uuid), join(INPUT_DIR, COLLECTION_ID, uuid, OCR_NAME),df,DEFAULT_FIELDS)
            for uuid in uuids
        )
        print(results)


import logging
import glob
from convert import check_gt_and_generate_copynet_file, STATUS
import os
from os.path import join
import pandas as pd
import sys
from joblib import Parallel, delayed
from jsonargparse import ArgumentParser
import json
import subprocess
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
DEFAULT_CONFIG = join(INPUT_DIR, "default_copynet.json")
MODEL = "model"

DEFAULT_FIELDS = {
    "ClaimNumber",
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
    "Address1",
}


def get_config_parser():

    parser = ArgumentParser()
    # TODO
    return parser

def update_default_config(default_path, train_path, val_path, max_target_length, intermediate_path):

    d = json.load(open(default_path))
    d["train_data_path"] = train_path
    d["validation_data_path"] = val_path
    d["model"]["max_decoding_steps"] = max_target_length + 1 
    json.dump(d,open(intermediate_path,"w"),ensure_ascii=False, indent=4)
    logger.info(f"dumped copynet config to {intermediate_path}")

def generate_copynet_files():
    max_source_length, max_target_length = 0,0
    splits = {}
    df = pd.read_csv(join(INPUT_DIR, CSV), dtype=str)
    for split, file in SPLITS.items():
        with open(join(INPUT_DIR, file)) as f:
            uuids = f.read().splitlines()
            results = Parallel(n_jobs=1)(
                delayed(check_gt_and_generate_copynet_file)(
                    join(OUTPUT_DIR, INTERMEDIATE, uuid),
                    join(INPUT_DIR, COLLECTION_ID, uuid, OCR_NAME),
                    df,
                    DEFAULT_FIELDS,
                )
                for uuid in uuids
            )
            ok_count = 0
            split_file = join(OUTPUT_DIR, INTERMEDIATE, split + ".txt")
            splits[split]["copynet"] = split_file
            splits[split]["paths"] = []
            with open(split_file, "w") as f:
                for res in results:
                    if res[1] == STATUS.OK:
                        ok_count += 1
                        max_source_length = max(max_source_length, res[0][1])
                        max_target_length = max(max_target_length, res[0][2])
                        splits[split]["paths"].append(  os.path.basename(os.path.dirname(result[0][0])) )
                        with open(res[0][0]) as g:
                            line = g.read()
                            f.write(line)
            logger.info(f"{ok_count} ok of {len(uuids)} in {split} ")
            if ok_count == 0:
                raise ValueError(f"Empty split {split}")
    logger.info(f"max_source_length: {max_source_length}, max_target_length {max_target_length}")
    return splits, max_source_length, max_target_length

def train_model(cfg_path,model_dir):
    process = subprocess.Popen(["allennlp", "train", cfg_path, "--serialization-dir", model_dir])
    code = process.wait()
    success = code == 0
    if success:
        logger.info("Model training successfull.")
    else:
        logger.info("Model training failed.")
    return success

def predict(model_dir, test_path,output_file):

    process = subprocess.Popen(["allennlp", "predict",  model_dir, test_path, "--cuda-device", "0", "--use-dataset-reader","--output-file", output_file])
    code = process.wait()
    success = code == 0
    if success:
        logger.info("Model prediction successfull.")
    else:
        logger.info("Model prediction failed.")
    return success


def convert_prediction_to_jsons(prediction_path,):
    return

if __name__ == "__main__":

    splits, max_source_length, max_target_length = generate_copynet_files()
    cfg_path = join(OUTPUT_DIR, INTERMEDIATE, "config.json")
    update_default_config(DEFAULT_CONFIG, splits["train"]["copynet"], splits["val"]["copynet"], max_target_length, cfg_path)
    model_dir = join(OUTPUT_DIR, MODEL)
    success =train_model(cfg_path, model_dir)
    if not success:
        exit
    prediction_path = join(OUTPUT_DIR, INTERMEDIATE, "test_predictions.txt")
    success = predict( model_dir, splits["test"]["copynet"], prediction_path)
    if not success:
        exit
    
    
    
    
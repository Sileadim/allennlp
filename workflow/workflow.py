# Worflow does the following:
# Check if experiment already exists
# Read collection dir and csv
# Read split files and create copynet files
# Run allennlp with provided config
# Predict on test set
# Convert test set to jsons
# Run mleval on predicted jsons

import math
import logging
import glob
from convert import check_gt_and_generate_copynet_file, STATUS
import os
from os.path import join
import pandas as pd
import sys
from joblib import Parallel, delayed
from jsonargparse import ArgumentParser, ActionPath, Path
import json
import subprocess
from allennlp.training.metrics.information_extraction import parse_json
from mlevaluation.information_extraction import InformationExtractionEvaluator
import shutil


ALL_FIELDS = [
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
]


DEFAULT_FIELDS = [
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
]


class Workflow:
    @staticmethod
    def get_config_parser():

        parser = ArgumentParser(env_prefix="APP", default_env=True)
        parser.add_argument("--experiment_name", type=str, help="Name of experiment folder")
        parser.add_argument(
            "--existing_intermediate_dir",
            type=str,
            help="Already existing intermediate path. Passing this will copy the intermediate files from there.",
        )
        parser.add_argument(
            "--output_dir",
            action=ActionPath(mode="dw"),
            help="Where to write intermediate and output files",
        )
        parser.add_argument(
            "--collection",
            action=ActionPath(mode="dr"),
            help="path to folder containing all the uuids",
        )
        parser.add_argument(
            "--splits.train", action=ActionPath(mode="fr"), help="list file with train uuids"
        )
        parser.add_argument(
            "--splits.val", action=ActionPath(mode="fr"), help="list file with val uuids"
        )
        parser.add_argument(
            "--splits.test", action=ActionPath(mode="fr"), help="list file with test uuids"
        )
        parser.add_argument("--csv", action=ActionPath(mode="fr"), help="path to gt csv file")
        parser.add_argument(
            "--default_copynet_config",
            action=ActionPath(mode="fr"),
            default=Path("default_copynet.json"),
            help="path to copynet default config",
        )
        parser.add_argument("--ocr_name", type=str, default="ocr-page.xml", help="ocr xml name")
        parser.add_argument(
            "--fields",
            nargs="+",
            choices=ALL_FIELDS,
            default=DEFAULT_FIELDS,
            help="which fields to use",
        )
        parser.add_argument("--recover", action="store_true", help="recover copynet training")
        parser.add_argument(
            "--skip_training",
            action="store_true",
            help="skip training step (only makes sense if there is already a trained model",
        )
        parser.add_argument("--skip_prediction", action="store_true", help="skip prediction step")
        parser.add_argument("--n_jobs", type=int, help="Number of joblib threads.")
        # exposing the 2 most important params
        parser.add_argument("--copynet.batch_size", type=int, help="Batch size", default=1)
        parser.add_argument("--copynet.epochs", type=int, help="Epochs", default=1)
        parser.add_argument("--only_alphanumeric", action="store_true", help="use only alphanumeric in gt and ocr")
        parser.add_argument("--lowercase", action="store_true", help="cast gt and ocr to lowercase")
        parser.add_argument("--max_token_length", action="store_true", help="cast gt and ocr to lowercase")

        return parser

    def __init__(self, cfg):

        self.cfg = cfg
        if not self.cfg.experiment_name:
            raise ValueError("No experiment name provided")
        self.experiment_dir = join(self.cfg.output_dir.path, self.cfg.experiment_name)

        if os.path.isdir(self.experiment_dir) and not (
            self.cfg.skip_prediction or self.cfg.skip_training or self.cfg.recover
        ):
            raise ValueError(
                f"Experiment {self.cfg.experiment_name} already exists and no flag indicating the continuing of an already existing experiments is chosen. Aborting"
            )
        else:
            os.makedirs(self.experiment_dir, exist_ok=True)
            parser = Workflow.get_config_parser()
            string_repr = parser.dump(self.cfg)
            with open(join(self.experiment_dir, "workflow_config.json"), "w") as f:
                f.write(string_repr + "\n")

        self.logger = logging.getLogger("Logger")
        formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
        self.logger.setLevel(logging.INFO)
        self.log_file = join(self.experiment_dir, "log.txt")
        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s : %(message)s",
            level=logging.INFO,
            stream=sys.stdout,
        )
        self.intermediate_dir = join(self.experiment_dir, "intermediate")
        if self.cfg.existing_intermediate_dir:
            shutil.copytree(self.cfg.existing_intermediate_dir, self.intermediate_dir)

        self.data_dir = join(self.intermediate_dir, "data")
        self.model_dir = join(self.experiment_dir, "model")
        self.copynet_prediction_path = join(self.intermediate_dir, "test_predictions.txt")
        self.report_dir = join(self.experiment_dir, "report")

    def run(self):

        self.generate_copynet_files()
        self.update_default_config()
        if not self.cfg.skip_training:
            success = self.train_model()
            if not success:
                return
        else:
            self.logger.info("skipping training")
        if not self.cfg.skip_prediction:
            success = self.predict()
            if not success:
                return
        else:
            self.logger.info("skipping predictions")
        self.convert_prediction_to_jsons()
        self.run_evaluation()

    def generate_copynet_files(self):
        max_source_length, max_target_length = 0, 0
        splits = {}
        df = pd.read_csv(self.cfg.csv.path, dtype=str)
        SPLITS = {
            "train": self.cfg.splits.train.path,
            "val": self.cfg.splits.val.path,
            "test": self.cfg.splits.test.path,
        }
        for split, file in SPLITS.items():
            with open(join(file)) as f:
                uuids = f.read().splitlines()
                results = Parallel(n_jobs=self.cfg.n_jobs)(
                    delayed(check_gt_and_generate_copynet_file)(
                        join(self.data_dir, uuid),
                        join(self.cfg.collection.path, uuid, self.cfg.ocr_name),
                        df,
                        self.cfg.fields,
                        lowercase=self.cfg.lowercase,
                        only_alphanumeric=self.cfg.only_alphanumeric
                    )
                    for uuid in uuids
                )
                ok_count = 0
                loaded_count = 0
                too_long = 0
                split_file = join(self.intermediate_dir, split + ".txt")
                splits[split] = {}
                splits[split]["copynet"] = split_file
                splits[split]["paths"] = []
                with open(split_file, "w") as f:
                    for res in results:
                        if res[1] in [STATUS.OK, STATUS.LOADED]:
                            if not self.cfg.max_token_length or res[0][1] <= self.cfg.max_token_length:
                                if res[1] == STATUS.OK:
                                    ok_count += 1
                                else:
                                    loaded_count += 1

                                max_source_length = max(max_source_length, res[0][1])
                                max_target_length = max(max_target_length, res[0][2])
                                # get uuid back
                                splits[split]["paths"].append(
                                    os.path.basename(os.path.dirname(res[0][0]))
                                )
                                with open(res[0][0]) as g:
                                    line = g.read()
                                    f.write(line)
                            else:
                                too_long += 1
                self.logger.info(
                    f"{ok_count} extract, {too_long} too_long, {loaded_count} loaded of {len(uuids)} in {split} "
                )
                if ok_count == 0 and loaded_count == 0:
                    raise ValueError(f"Empty split {split}")
        self.logger.info(
            f"max_source_length: {max_source_length}, max_target_length {max_target_length}"
        )

        self.splits, self.max_source_length, self.max_target_length = (
            splits,
            max_source_length,
            max_target_length,
        )

    def update_default_config(self):

        d = json.load(open(self.cfg.default_copynet_config.path))
        d["iterator"]["batch_size"] = self.cfg.copynet.batch_size
        d["trainer"]["num_epochs"] = self.cfg.copynet.epochs
        d["train_data_path"] = self.splits["train"]["copynet"]
        d["validation_data_path"] = self.splits["val"]["copynet"]
        # give 10 percent slack, so that if we copy more for one field the decoding does not stop
        d["model"]["max_decoding_steps"] = math.floor(self.max_target_length * 1.1)
        out_cfg_path = join(self.intermediate_dir, "config.json")
        self.copynet_config = out_cfg_path
        json.dump(d, open(out_cfg_path, "w"), ensure_ascii=False, indent=4)
        self.logger.info(f"dumped copynet config to {out_cfg_path}")

    def train_model(self):
        self.logger.info("Starting training, redirecting allennlp output to log file.")
        was_killed = False
        try:
            args = ["allennlp", "train", self.copynet_config, "--serialization-dir", self.model_dir]
            if self.cfg.recover:
                args.append("--recover")
            process = subprocess.Popen(
                args, stdout=open(self.log_file, "a"), stderr=open(self.log_file, "a")
            )
            code = process.wait()
        except KeyboardInterrupt:
            process.kill()
            code = 0
            was_killed = True
        success = code == 0
        if success:
            if was_killed:
                self.logger.info("Model training was manually terminated.")
            else:
                self.logger.info("Model training successfull.")
        else:
            self.logger.error("Model training failed.")
        return success

    def predict(self):

        self.logger.info("Starting prediction, redirecting allennlp output to log file.")
        process = subprocess.Popen(
            [
                "allennlp",
                "predict",
                self.model_dir,
                self.splits["test"]["copynet"],
                "--cuda-device",
                "0",
                "--use-dataset-reader",
                "--output-file",
                self.copynet_prediction_path,
            ],
            stdout=open(self.log_file, "a"),
            stderr=open(self.log_file, "a"),
        )
        code = process.wait()
        success = code == 0
        if success:
            self.logger.info("Model prediction successfull.")
        else:
            self.logger.error("Model prediction failed.")
        return success

    def convert_prediction_to_jsons(self):

        predicted_json_paths = []
        uuids = self.splits["test"]["paths"]
        with open(self.copynet_prediction_path, "r") as f:
            preds = f.read().splitlines()
            assert len(preds) == len(uuids)
            for pred, uuid in zip(preds, uuids):
                prediction_dict = json.loads(pred)
                predicted_json = parse_json(prediction_dict["predicted_tokens"][0], [], False)
                import pdb; pdb.set_trace()
                out_path = join(self.data_dir, uuid, "pred.json")
                json.dump(predicted_json, open(out_path, "w"), ensure_ascii=False, indent=4)
                predicted_json_paths.append(out_path)
        self.logger.info("Converted to proper jsons.")
        self.predicted_json_paths = predicted_json_paths

    def run_evaluation(self):

        parser = InformationExtractionEvaluator.get_config_parser()
        cfg = parser.parse_args(["--doc_id", "parent-dir"])
        evaluator = InformationExtractionEvaluator(cfg=cfg)
        gt_json_paths = [pred.replace("pred", "gt") for pred in self.predicted_json_paths]
        evaluator.generate_report(
            outdir=self.report_dir, gt=gt_json_paths, pr=self.predicted_json_paths
        )
        self.logger.info("Evaluation done")


if __name__ == "__main__":

    parser = Workflow.get_config_parser()
    cfg = parser.parse_args(env=True)
    workflow = Workflow(cfg=cfg)
    workflow.run()

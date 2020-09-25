from overrides import overrides
import os
from allennlp.training.metrics.metric import Metric
import json
import re
from collections import OrderedDict
from mlevaluation.information_extraction import InformationExtractionEvaluator
import tempfile
import traceback
# need to parse args, no other way to input args
parser = InformationExtractionEvaluator.get_config_parser()
cfg = parser.parse_args(["--doc_id", "parent-dir"])
evaluator = InformationExtractionEvaluator(cfg=cfg)

token = re.compile("@@[0-9A-Za-z_{}[\]\.,]+@@")


def get_matches(pred_string):
    matches = token.finditer(pred_string)

    def clean(string):
        return string.replace("@@", "")

    span_text_type = []
    for match in matches:
        if not any({x in match.group(0) for x in ["{", "}", "]", "[", ","]}):
            type = "key"
        elif "," in match.group(0):
            type = "comma"
        else:
            if any({x in match.group(0) for x in ["{", "}"]}):
                type = "curly"
            else:
                type = "square"

        span_text_type.append(
            {"start": match.span()[0], "end": match.span()[1], "text": clean(match.group(0)), "type": type})
    return span_text_type


def format_text_field(string, fix_spaces=False):
    string = string.strip(" ")
    if string == "null":
        return string
    # quote " chars
    string = string.replace('"', '\\"')

    if fix_spaces:
        for punct in [";", ",", ".", "!"]:
            string = string.replace(" " + punct, punct)
        for bracket in ["(", "["]:
            string = string.replace(bracket + " ", bracket)
        for bracket in ["]", ")"]:
            string = string.replace(" " + bracket, bracket)
        string = string.replace("- ", "-").replace(" -", "-")
    return '"' + string + '"'


def parse_json(pred, recurring_list=[], fix_spaces=False):
    pred_string = " ".join(pred)
    # first replace null tokens because we don't want to treat them like brackets and keys
    pred_string = pred_string.replace("@@UNKNOWN@@", "").replace("@@null@@", "null")
    span_text_type = get_matches(pred_string)
    print(pred_string)
    #import pdb; pdb.set_trace()
    def next_start(i):
        if (i + 1) == len(span_text_type):
            return i + 1
        else:
            return span_text_type[i + 1]["start"]

    def is_next_type(i, type="key"):
        if (i + 1) == len(span_text_type):
            return False
        else:
            return span_text_type[i + 1]["type"] == type

    def is_next_text(i, text):
        if (i + 1) == len(span_text_type):
            return False
        else:
            return span_text_type[i + 1]["text"] == text

    def is_next_open(i):
        if (i + 1) == len(span_text_type):
            return False
        else:
            return any([x in span_text_type[i + 1]["text"] for x in ["{", "["]])

    cleaned_string_lst = []
    # we try to produce a proper json by keeping track of brackets
    bracket_stack = []
    def update_stack(i, match):

        if match["type"] == "curly":
            if match["text"] == "{":
                bracket_stack.append("{")
                cleaned_string_lst.append("{")
            # closing curly bracket
            else:
                # here we we would add a curly closing without any opening so let's skip this
                if len(bracket_stack) == 0:
                    pass
                #    print("Should not happen: ", bracket_stack, match["text"])
                # if the previous one is a opening one, pop from stack
                elif bracket_stack[-1] == "{":
                    _ = bracket_stack.pop()
                    cleaned_string_lst.append("}")
                    # we assume if the next one is a curly bracket we add a comma
                    if is_next_text(i, "{") or is_next_type(i, "key"):
                        cleaned_string_lst.append(",")
                # this also should not happen, so insert ] before, and pop [ from stack. Then we need to check again
                # so we call update_stack
                elif bracket_stack[-1] == "[":

                    #print("Should not happen: ", bracket_stack, match["text"])
                    #print("Trying to fix")
                    _ = bracket_stack.pop()
                    cleaned_string_lst.append("]")
                    update_stack(i, match)
                # this case should not happen, because we caught it when checking for square
                elif bracket_stack[-1] == "]":
                    print("Should not happen: ", bracket_stack, match["text"])
        # square brackets
        elif match["type"] == "square":
            if match["text"] == "[":
                bracket_stack.append("[")
                cleaned_string_lst.append("[")
            # closing square
            else:
                # here we we would add a square closing without any opening so let's skip this
                if len(bracket_stack) == 0:
                    print("Should not happen: ", bracket_stack, match["text"])
                # if the previous one is a opening one, pop from stack
                elif bracket_stack[-1] == "[":
                    _ = bracket_stack.pop()
                    cleaned_string_lst.append("]")
                    # if next is a new list or a key, add comma
                    if is_next_text(i, "[") or is_next_type(i, "key"):
                        cleaned_string_lst.append(",")
                # this also should not happen, so insert ] before, and pop [ from stack. Then we need to check again
                # so we call update_stack
                elif bracket_stack[-1] == "{":
                    _ = bracket_stack.pop()
                    cleaned_string_lst.append("}")
                    update_stack(i, match)
                # this case should not happen, because we caught it when checking for curly
                elif bracket_stack[-1] == "}":
                    print("Should not happen: ", bracket_stack, match["text"])

    for i, match in enumerate(span_text_type):
        # if it is a key, add the key with '
        if match["type"] == "key":
            cleaned_string_lst.append('"' + match["text"] + '":')
            # if it's not a dict or list, add the next string between matches of tokens
            if match["text"] not in recurring_list:
                if not is_next_open(i):
                    cleaned_string_lst.append(format_text_field(pred_string[match["end"]+1:next_start(i)], fix_spaces)
                                              )
                # if next is a key, we need to add a comma
                if is_next_type(i, "key"):
                    # add text between tokens
                    cleaned_string_lst.append(",")
        if match["type"] == "comma":
            cleaned_string_lst.append(",")
            if pred_string[match["end"]+1:next_start(i)]:
                cleaned_string_lst.append(format_text_field(pred_string[match["end"]+1:next_start(i)], fix_spaces))

        elif match["text"] == "[" and not is_next_open(i):
            update_stack(i, match)
            cleaned_string_lst.append(format_text_field(pred_string[match["end"]:next_start(i)], fix_spaces))

        # if it is not a key, we can ignore the following stuff
        else:
            update_stack(i, match)

    # fix missing brackets in the end
    for b in reversed(bracket_stack):
        if b == "{":
            cleaned_string_lst.append("}")
        if b == "[":
            cleaned_string_lst.append("]")

    cleaned_string = " ".join(cleaned_string_lst)
    print(cleaned_string)
    try:
        d = json.loads(cleaned_string, object_pairs_hook=OrderedDict)
    except Exception as e:
        #print(pred_string)
        #traceback.print_exc()
        d = {}
    return d


def filter_double_recurring(d, recurring_list):
    for recurring in recurring_list:
        # only check on 1st level, so we assume a list of (lists or dicts) with string values
        if recurring in d.keys():
            new_list = []
            recurring_items = d[recurring]
            # no sets so we keep order; only have
            for r in recurring_items:
                # also remove empty ones
                if r and r not in new_list:
                    new_list.append(r)

            d[recurring] = new_list


@Metric.register("information_extraction")
class InformationExtraction(Metric):
    """
        Mleval information extraction as a metric
    """

    def __init__(self) -> None:
        self.gt_jsons = []
        self.pred_jsons = []

    def __call__(
            self,
            predictions,
            gt,
    ):
        """
        # Parameters

        predictions : `List[List[str]`, required.
            Gt tokens
        gt : `List[List[str]`, required.
            Predicted tokens

        """

        self.gt_jsons += [parse_json(gt_list) for gt_list in gt]
        self.pred_jsons += [parse_json(pred_list) for pred_list in predictions]

    @overrides
    def reset(self):
        self.gt_jsons = []
        self.pred_jsons = []

    def get_metric(self, reset: bool = False):
        """
         Returns:
            dict: f1, recall and precision

        """

        tmp_pred = []
        tmp_gt = []
        if any(self.pred_jsons):
            with tempfile.TemporaryDirectory() as tmpdirname:
                for i, (pred, gt) in enumerate(zip(self.pred_jsons, self.gt_jsons)):
                    subdir = os.path.join(tmpdirname, str(i))
                    os.makedirs(subdir)
                    gt_path = os.path.join(subdir, "gt.json")
                    pred_path = os.path.join(subdir, "pred.json")
                    json.dump(pred, open(pred_path, "w"), ensure_ascii=False)
                    json.dump(gt, open(gt_path, "w"), ensure_ascii=False)
                    tmp_pred.append(pred_path)
                    tmp_gt.append(gt_path)
                try:
                    out = evaluator.evaluate(tmp_gt, tmp_pred)
                    if reset:
                        self.reset()

                    return {"f1": out["overall"]["f1"][0], "recall": out["overall"]["recall"][0],
                            "precision": out["overall"]["precision"][0]}
                except:
                    pass
        if reset:
            self.reset()

        return {"f1": 0, "recall": 0,
                "precision": 0}


if __name__ == "__main__":
    import sys

    parse_json(sys.argv[1].split())

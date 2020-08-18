from jsonargparse import ArgumentParser, ActionConfigFile
import os
from os.path import join
import json
from pagexmltools.process import page_region_with_ordered_textlines, merge_gt_and_ocr
from joblib import Parallel, delayed
import spacy
import copy
from collections import OrderedDict
from enum import Enum
import pagexml
import numpy as np

def process_coords(coords):
    """Converts pagexml points object into list of xmin, ymin, xmax, ymax


    Arguments:
        coords {pagexml point object} -- A pagexml points object

    Returns:
        list -- List of integers: xmin,ymin,xmax,ymax, assuming top left is 0,0.
    """

    return [coords[0].x, coords[0].y, coords[2].x, coords[2].y]


def normalize_to_page_dim(coords, width, height):
    """ Takes coordinates in the xmin, ymin, xmax, ymax format and normalizes them inplace
        by dividing the x's by the width and y's by the height.

    Arguments:
        coords {list} -- List of integers: xmin,ymin,xmax,ymax, assuming top left is 0,0.
        width {float} -- page width
        height {float} -- page height
    """

    coords[0] = coords[0] / width
    coords[2] = coords[2] / width
    coords[1] = coords[1] / height
    coords[3] = coords[3] / height


def coords_to_str(coords):
    return ",".join(str(c) for c in coords)


def get_words_coords_size_style_tags_from_pxml(
    pxml,
    ignore_tags=False,
    ignore_size=False,
    ignore_style=False,
    page_selector="//_:Page",
    keep_textline=False,
    add_page_number=False,
):
    """
    Arguments:
        pxml {PageXML object} -- PageXML object

    Keyword Arguments:
        ignore_tags {bool} -- If true, no tags are extracted and returned
        ignore_size {bool} -- If true, no sizes are extracted and returned
        ignore_style {bool} -- If true, no styles are extracted and returned
        page_selector {str} -- XML expression defining the path (default: '//_:Page')

    Returns:
        list -- List of words for each page
        list -- List of coords for each page
        list -- List of tags for each page (not returned if ignore_tags=True)
        list -- List of font sizes for each page (not returned if ignore_size=True)
        list -- List of font styles for each page (not returned if ignore_style=True)

    """

    all_words = []
    all_coords = []
    all_tags = []
    all_sizes = []
    all_styles = []
    # Reorder lines for a top to bottom, left to right reading order
    page_region_with_ordered_textlines(pxml, fake_baseline=True)

    # Select pages according to provided page selector
    for page_number, page in enumerate(pxml.select(page_selector)):

        # Get width and height for normalization
        page_width = pxml.getPageWidth(page)
        page_height = pxml.getPageHeight(page)

        words = []
        coords = []
        tags = []
        styles = []
        sizes = []
        # get all words
        for textline in pxml.select(".//_:TextLine", page):

            words_per_textline = []
            coords_per_textline = []
            tags_per_textline = []
            styles_per_textline = []
            sizes_per_textline = []

            for word in pxml.select(".//_:Word", textline):

                text_equiv = pxml.getTextEquiv(word)
                words_per_textline.append(text_equiv)
                # get and process coordinates
                word_coords = process_coords(pxml.getPoints(word))
                normalize_to_page_dim(word_coords, page_width, page_height)
                if add_page_number:
                    word_coords.append(page_number)
                coords_per_textline.append(word_coords)

                if not ignore_tags:
                    tag = pxml.getPropertyValue(word, "entity")
                    tags_per_textline.append(tag)
                if not ignore_style:
                    style = pxml.getPropertyValue(word, "style")
                    styles_per_textline.append(style)
                if not ignore_size:
                    font_size = pxml.getPropertyValue(word, "font-size")
                    sizes_per_textline.append(font_size)

            if not keep_textline:
                words += words_per_textline
                coords += coords_per_textline
                if not ignore_tags:
                    tags += tags_per_textline
                if not ignore_style:
                    styles += styles_per_textline
                if not ignore_size:
                    sizes += sizes_per_textline

            else:
                words.append(words_per_textline)
                coords.append(coords_per_textline)
                if not ignore_tags:
                    tags.append(tags_per_textline)
                if not ignore_style:
                    styles.append(styles_per_textline)
                if not ignore_size:
                    sizes.append(sizes_per_textline)

        all_words.append(words)
        all_coords.append(coords)
        all_tags.append(tags)
        all_sizes.append(sizes)
        all_styles.append(styles)

    return_list = [all_words, all_coords]

    if not ignore_size:
        return_list.append(all_sizes)
    if not ignore_style:
        return_list.append(all_styles)
    if not ignore_tags:
        return_list.append(all_tags)

    return tuple(return_list)


def extract_ocr(pxml, page_selector="//_:Page", add_page_number=False):

    words_and_coords = get_words_coords_size_style_tags_from_pxml(pxml, page_selector=page_selector)

    only_words = words_and_coords[0]
    coordinates_per_page = words_and_coords[1]
    text = []
    coords_text = []
    for p in only_words:
        if p:
            text.append(" ".join(p))

    for page_number, coordinates in enumerate(coordinates_per_page):
        if coordinates:
            if add_page_number:
                coords_text.append(
                    " ".join(
                        [coords_to_str(coords) + "," + str(page_number) for coords in coordinates]
                    )
                )
            else:
                coords_text.append([coords_to_str(coords) for coords in coordinates])

    full_text = " ".join(text)
    full_coords_text = " ".join(coords_text)

    assert len(full_coords_text.split()) == len(full_text.split())

    if not full_text:
        full_text = "@@empty@@"
        full_coords_text = "0.0,0.0,0.0,0.0"
        if add_page_number:
            full_coords_text = full_coords_text + ",0"
    return full_text, full_coords_text


def generate_repr(obj):
    list_of_repr = []
    if isinstance(obj, dict):
        list_of_repr.append("@@{@@")
        for i, (key, val) in enumerate(obj.items()):
            list_of_repr.append(f"@@{key}@@")
            list_of_repr.append(generate_repr(val))
        list_of_repr.append("@@}@@")
    elif isinstance(obj, list):
        list_of_repr.append("@@[@@")
        for i, o in enumerate(obj):
            list_of_repr.append(generate_repr(o))
            if not (any(isinstance(o, x) for x in [list, dict])) and i < len(obj) - 1:
                list_of_repr.append("@@,@@")
        list_of_repr.append("@@]@@")
    else:
        if obj is None:
            list_of_repr.append("@@null@@")
        else:
            list_of_repr.append(obj)

    return " ".join(list_of_repr)


def generate_learnable_dump(json_path):
    json_dict = json.load(open(json_path, "r"))
    return generate_repr(json_dict)




class STATUS(Enum):
    NO_GT_IN_CSV = 1
    CORRUPT_XML = 2
    EXTRACTION_ERROR = 3
    DUMP_GENERATION_ERROR = 4
    WRITE_ERROR = 5
    OK = 6

import pandas as pd
import json
def emptyToNone(value):
    #print(value, pd.isnull(value))
    if pd.isnull(value) or value == "":
        return None
    return value

def check_gt_and_generate_copynet_file(
    out_dir, ocr_path, df, extract_keys, page_selector="//_:Page", add_page_number=True, tokenizer=None
):

    try:
        pxml = pagexml.PageXML(ocr_path)
        doc_node = pxml.select("//_:PcGts")[0]
        fileName = pxml.getPropertyValue(doc_node, "fileName")
    except Exception as e:
        return (None, STATUS.CORRUPT_XML)
    if not fileName in list(df["FileName"]):
        return (None, STATUS.NO_GT_IN_CSV)
    row_dict = df.loc[df['FileName'] == fileName].to_dict()
    new_dict = {key: emptyToNone(list(value.values())[0]) for key, value in row_dict.items() if key in extract_keys}
    try:
        full_text, full_coords_text = extract_ocr(pxml, page_selector, add_page_number)
    except:
        return (None, STATUS.EXTRACTION_ERROR)
    try:
        dump = generate_repr(new_dict)
    except Exception as e:
        return (None, STATUS.DUMP_GENERATION_ERROR)
    try:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(new_dict, open(os.path.join(out_dir, "gt.json"), "w"), ensure_ascii=False, indent=4)
        copynet_file = os.path.join(out_dir, "copynet.tsv")
        with open(copynet_file, "w") as f:
            f.write("\t".join([full_text, full_coords_text, dump]) + "\n")
    except:
        return (None, STATUS.WRITE_ERROR)
    if tokenizer:
        nlp = spacy.load(tokenizer)
        return len(nlp(full_text)), len(nlp(dump))
    return ((copynet_file, len(full_text.split()), len(dump.split())), STATUS.OK)


if __name__ == "__main__":
    parser = ArgumentParser(description="Parse training parameters")
    parser.add_argument("path_to_dataset", type=str, help="path where to write data files")

    parser.add_argument(
        "--path_to_uuid_root", type=str, help="the where to read gt files from", default=None
    )
    parser.add_argument("--list_of_uuid_dirs", type=str, default=None)

    parser.add_argument("--ocr", type=str, default="ocr_abbyy.xml", help="ocr file name")

    parser.add_argument("--gt_json", type=str, default="gt.json", help="ocr file name")
    parser.add_argument("--gt_xml", type=str, default="page.xml", help="ocr file name")
    parser.add_argument("--gt_model_json", type=str, default="gt_model.json", help="gt_model.json")

    group1 = parser.add_mutually_exclusive_group()

    parser.add_argument("--page_xpath", type=str, default="//_:Page")
    parser.add_argument("--add_page_number", action="store_true")

    parser.add_argument("--model_json", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)

    args = parser.parse_args()
    print(args)
    generate_dataset(
        args.path_to_dataset,
        args.path_to_uuid_root,
        args.list_of_uuid_dirs,
        args.ocr,
        args.gt_json,
        args.gt_xml,
        args.page_xpath,
        args.model_json,
        args.add_page_number,
        args.tokenizer,
        args.gt_model_json,
    )


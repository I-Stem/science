"""
Helper file for extract the table from the image/scaned documents
"""

import os
import io
import sys
import cv2
import json
import boto3
import pickle
import base64
import requests
import tempfile
import webbrowser
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from pprint import pprint
from .error_handler import TableExtractionError
from pdf2image import convert_from_path, convert_from_bytes

def get_table_csv_results(
    file_name: str, save_name: str = None, ocr_saver: str = None
) -> list:
    """
    If image has image, the detect the image cell wise.
    
    """
    os.environ[os.getenv('TABLE_AUTH_KEY1')] = os.getenv("TABLE_AUTH_VALUE1")
    os.environ[os.getenv('TABLE_AUTH_KEY2')] = os.getenv("TABLE_AUTH_VALUE2")

    try:
        # with open(file_name, 'rb') as file:
        # img_test = file.read()
        img_test = file_name
        bytes_test = bytearray(img_test)

        client = boto3.client(os.getenv("TABLE_NAME"), os.getenv("TABLE_REGION"))
        response = client.analyze_document(
            Document={"Bytes": bytes_test}, FeatureTypes=["TABLES"]
        )
        print("TableExtraction: got result of document analysis")
        blocks = response["Blocks"]

        if ocr_saver != None:
            with open(ocr_saver, "wb") as f:
                pickle.dump(blocks, f)

        blocks_map = {}
        table_blocks = []
        for block in blocks:
            blocks_map[block["Id"]] = block
            if block["BlockType"] == "TABLE":
                table_blocks.append(block)

        if len(table_blocks) <= 0:
            print("Table Extraction: NO Table FOUND")
            return []

        csv = []
        for index, table in enumerate(table_blocks):
            z = table["Geometry"]["BoundingBox"]
            _table = table_to_dataframe(table, blocks_map, index + 1)
            if save_name != None:
                file_name = save_name.replace(".csv", f"_{str(index)}.csv")
                _table.to_csv(file_name)

            tmp = dict(
                label="table",
                relative_box=[
                    z["Left"],
                    z["Top"],
                    z["Left"] + z["Width"],
                    z["Height"] + z["Top"],
                ],
                text=_table,
            )
            csv.append(tmp)
        return csv
    except TableExtractionError as ex:
        print("Table extraction: error detected")
        print(ex)
        return []


def get_text(result, blocks_map):
    """
    helper function to extract the table data
    """
    text = ""
    if "Relationships" in result:
        for relationship in result["Relationships"]:
            if relationship["Type"] == "CHILD":
                for child_id in relationship["Ids"]:
                    word = blocks_map[child_id]
                    if word["BlockType"] == "WORD":
                        text += word["Text"] + " "
                    if word["BlockType"] == "SELECTION_ELEMENT":
                        if word["SelectionStatus"] == "SELECTED":
                            text += "X "
    return text


def get_rows_columns_map(table_result, blocks_map):
    """
    Helper function for extracting the row wise data.
    """
    rows = {}
    for relationship in table_result["Relationships"]:
        if relationship["Type"] == "CHILD":
            for child_id in relationship["Ids"]:
                cell = blocks_map[child_id]
                if cell["BlockType"] == "CELL":
                    row_index = cell["RowIndex"]
                    col_index = cell["ColumnIndex"]
                    if row_index not in rows:
                        # create new row
                        rows[row_index] = {}

                    # get the text value
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows


def table_to_dataframe(table_result, blocks_map, table_index):
    """
    convert list of list into pandas dataframe.
    """
    rows = get_rows_columns_map(table_result, blocks_map)
    table_data = pd.DataFrame(rows).T
    return table_data

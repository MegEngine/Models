# Copyright (c) 2020 Kwot Sin Lee
# This code is licensed under MIT license
# (https://github.com/kwotsin/mimicry/blob/master/LICENSE)
# ------------------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
# ------------------------------------------------------------------------------
"""
Script for common utility functions.
"""
import json
import os

import numpy as np


def write_to_json(dict_to_write, output_file):
    """
    Outputs a given dictionary as a JSON file with indents.

    Args:
        dict_to_write (dict): Input dictionary to output.
        output_file (str): File path to write the dictionary.

    Returns:
        None
    """
    with open(output_file, 'w') as file:
        json.dump(dict_to_write, file, indent=4)


def load_from_json(json_file):
    """
    Loads a JSON file as a dictionary and return it.

    Args:
        json_file (str): Input JSON file to read.

    Returns:
        dict: Dictionary loaded from the JSON file.
    """
    with open(json_file, 'r') as file:
        return json.load(file)

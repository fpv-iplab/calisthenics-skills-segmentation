"""
Module: heur

This module contains functions for video segment reconstruction (heur).
"""

from statistics import mode
import pandas as pd
import os
import sys 

# Get the current script's file path
script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
parent_directory = os.path.dirname(script_directory)
sys.path.insert(1, parent_directory)

from inference_scripts.codec import encoding

def adjust_window_value(data):
    """
    Adjust the window value based on the provided data.

    Args:
        data (pandas.DataFrame): The input data for adjustment.

    Returns:
        int: The adjusted window value.
    """
    score_value = 0.5  # Initialize score_value in the range [0, 1]
    length = len(data)

    for i in range(length - 2):
        current_value = data.iloc[i]
        next_value = data.iloc[i + 1]

        if (current_value == next_value).all():
            score_value += 0.14 / length
        else:
            score_value -= 0.11 / length

        # Ensure score_value stays within the [0, 1] range
        score_value = max(0, min(1, score_value))
        reverse = 1 - score_value
        window_value = int(reverse * 32)

    return window_value

def swme(raw_predicted, window_size):
    """
    Calculate modes of data in windows.

    Args:
        raw_predicted (pandas.DataFrame): The input data.
        window_size (int): The size of the window for mode calculation.

    Returns:
        list: List of modes in windows.
    """
    modes = []
    local_mode = []
    ws = window_size
    i = window_size
    while i <= len(raw_predicted):
        values = raw_predicted[i - window_size:i].iloc[:, 0]
        current_mode = values.mode()[0]
        if len(set(values)) == window_size:
            window_size += 1
            i += 1
        else:
            min_index = values[values == current_mode].index.min()
            window_size = ws
            i += window_size - 3
            max_index = values[values == current_mode].index.max()

            if i >= len(raw_predicted):
                max_index = len(raw_predicted) - 1

            local_mode = [current_mode, min_index, max_index]
            modes.append(local_mode)

    return modes

def filter(pointer, modes):
    """
    Filter modes for noise removal.

    Args:
        pointer (int): The current pointer position.
        modes (list): List of modes to be filtered.

    Returns:
        list: Filtered modes.
    """
    patch_mode = []
    patch_mode.append(modes[pointer][0])

    if pointer - 1 >= 0:
        patch_mode.append(modes[pointer - 1][0])
    if pointer - 2 >= 0:
        patch_mode.append(modes[pointer - 2][0])
    if pointer + 1 < len(modes):
        patch_mode.append(modes[pointer + 1][0])
    if pointer + 2 < len(modes):
        patch_mode.append(modes[pointer + 2][0])

    new_moda = mode(patch_mode)
    modes[pointer][0] = new_moda
    return modes

def fnr(modes):
    """
    Perform noise removal on the list of modes.

    Args:
        modes (list): List of modes.

    Returns:
        list: Modes after noise removal.
    """
    if len(modes) > 3:
        for p in range(0, len(modes)):
            filter(p, modes)
    return modes

def tr(modes):
    """
    Reconstruct the timeline based on modes.

    Args:
        modes (list): List of modes.

    Returns:
        tuple: A tuple containing two lists, one with timeline data and the other with durations.
    """
    output_frame = []
    output_time = []
    j = 0
    breakp = False
    while j < len(modes) - 1:
        skill = modes[j][0]
        if j == 0:
            start = 0
        else:
            start = modes[j][1]

        while j < len(modes) - 1 and modes[j][0] == skill:
            j += 1

            if j == len(modes) - 1:
                end = modes[j][2]
                breakp = True

        if not breakp:
            end = modes[j][1] - 1

        output_frame.append([skill, start, end])
        milliseconds = ((end + 1) - start) * (1 / 24)
        output_time.append([skill, milliseconds])

    return output_frame, output_time

def heuristic_algorithm(raw_predicted, window_size=13):
    """
    Perform video segment reconstruction (heur) algorithm.

    Args:
        raw_predicted (pandas.DataFrame or list): The input data.
        window_size (int): The size of the window for mode calculation.

    Returns:
        tuple: A tuple containing the heur result, timeline data, and durations.
    """
    if isinstance(raw_predicted, list):
        raw_predicted = pd.DataFrame(raw_predicted)

    total_frames = len(raw_predicted)

    #Comment the following line to manually set the window size
    window_size = adjust_window_value(raw_predicted)

    if total_frames < window_size:
        window_size = total_frames
        return raw_predicted

    #Sliding Window Mode Extractor
    modes = swme(raw_predicted, window_size)
    #Filtering and Noise Removal
    modes = fnr(modes)
    #Timeline Reconstructor
    output_frame, output_time = tr(modes)

    heur_predicted = []
    for i in range(0, len(output_frame)):
        for j in range(output_frame[i][1], output_frame[i][2] + 1):
            heur_predicted.append(output_frame[i][0])

    heur_predicted = encoding(heur_predicted)

    return heur_predicted, output_frame, output_time
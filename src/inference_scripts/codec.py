def encoding(label_list):
    """
    Encode a list of string labels into numerical values.

    Parameters:
    - label_list (list of str): List of labels to be encoded.

    Returns:
    - list of int: List of encoded numerical values.
    """
    for i in range(len(label_list)):
        if label_list[i] == "bl":
            label_list[i] = 0
        elif label_list[i] == "fl":
            label_list[i] = 1
        elif label_list[i] == "flag":
            label_list[i] = 2
        elif label_list[i] == "ic":
            label_list[i] = 3
        elif label_list[i] == "mal":
            label_list[i] = 4
        elif label_list[i] == "none":
            label_list[i] = 5
        elif label_list[i] == "oafl":
            label_list[i] = 6
        elif label_list[i] == "oahs":
            label_list[i] = 7
        elif label_list[i] == "pl":
            label_list[i] = 8
        elif label_list[i] == "vsit":
            label_list[i] = 9
    return label_list

def decoding(encoded_list):
    """
    Decode a list of numerical values into corresponding string labels.

    Parameters:
    - encoded_list (list of int): List of numerical values to be decoded.

    Returns:
    - list of str: List of decoded string labels.
    """
    for i in range(len(encoded_list)):
        if encoded_list[i] == 0:
            encoded_list[i] = "bl"
        elif encoded_list[i] == 1:
            encoded_list[i] = "fl"
        elif encoded_list[i] == 2:
            encoded_list[i] = "flag"
        elif encoded_list[i] == 3:
            encoded_list[i] = "ic"
        elif encoded_list[i] == 4:
            encoded_list[i] = "mal"
        elif encoded_list[i] == 5:
            encoded_list[i] = "none"
        elif encoded_list[i] == 6:
            encoded_list[i] = "oafl"
        elif encoded_list[i] == 7:
            encoded_list[i] = "oahs"
        elif encoded_list[i] == 8:
            encoded_list[i] = "pl"
        elif encoded_list[i] == 9:
            encoded_list[i] = "vsit"
    return encoded_list

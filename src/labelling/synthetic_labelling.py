from typing import List, Dict
import re

# --- Extended label map ---
LABEL_MAP = {
    "O": 0,
    "B-HEADER": 1,
    "I-HEADER": 2,
    "B-TABLE": 3,
    "I-TABLE": 4,
}

import re


def synthetic_labeling(tokens: List[Dict]) -> List[int]:
    """
    Rule-based synthetic labeling:
    - Header tokens -> B-HEADER / I-HEADER
    - Table tokens -> B-TABLE / I-TABLE
    - All others -> O
    """
    labels = []
    prev_type = None  # track if previous token was header/table

    for t in tokens:
        is_header = bool(t.get("header", False))
        is_table = bool(t.get("inside_table", False))
        line_text = t.get("line_text")

        if is_table:
            curr_type = "TABLE"
        elif is_header:
            curr_type = "HEADER"
        else:
            curr_type = None

        if curr_type is None:
            labels.append(LABEL_MAP["O"])
        else:
            if curr_type != prev_type:
                labels.append(LABEL_MAP[f"B-{curr_type}"])
            else:
                labels.append(LABEL_MAP[f"I-{curr_type}"])

        prev_type = curr_type

    return labels

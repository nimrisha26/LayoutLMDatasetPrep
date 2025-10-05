import pytesseract
from PIL import Image as PILImage
from typing import List, Dict, Tuple
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
from img2table.document import Image as Img2TableImage
from img2table.ocr import DocTR as Img2TableOCR
import re

# load pretrained model
model = ocr_predictor(pretrained=True)

def ocr_pytesseract(image_path: str) -> List[Dict]:
    """
    Use pytesseract to extract word-level tokens + bboxes.
    Returns list of dicts: [{'word': str, 'bbox': (x0,y0,x1,y1)} ...]
    """
    # pytesseract image_to_data returns a TSV with bbox+text
    data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
    tokens = []
    n = len(data['level'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i]) if type(data['conf'][i]) == int else 0
        if not text:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        bbox = (x, y, x + w, y + h)
        if conf > 60 and (text.isupper() or h > 25):
           tokens.append({'word': text, 'bbox': bbox, "header": True})
        else:
           tokens.append({'word': text, 'bbox': bbox, "header": False})
    return tokens

def is_bold(image, bbox):
    cropped = image.crop(bbox)
    gray = cropped.convert("L")
    arr = np.array(gray)
    black_ratio = (arr < 128).mean()  # fraction of dark pixels
    return black_ratio > 0.15  # tune threshold

def check_headers(line_text: str, line_bbox, current_image) -> bool:
    """
    Detects if a given line of text is likely a section or subsection header.

    Rules:
    - Starts with a number (e.g. '1.', '2.1', '3)')
    - Starts with a Roman numeral (e.g. 'I.', 'II)', 'III-')
    - Starts with a capital letter followed by '.' or ')' (e.g. 'A.', 'B)')
    - Short lines (< 90 chars) with these patterns and some space after
    """

    if not line_text or len(line_text.strip()) < 2:
        return False

    text = line_text.strip()

    if is_bold(current_image, line_bbox) and  line_text[0].isupper():
        return True

    # --- Numeric headers (e.g., "1.", "2.3", "3)") ---
    if re.match(r'^\d+(\.\d+)*[\.\)]?\s', text):
        return True

    # --- Roman numeral headers (e.g., "I.", "II)", "IV-") ---
    if re.match(r'^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})'
                r'(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})[\.\)\-]\s', text, re.IGNORECASE):
        return True

    # --- Alphabetic headers (e.g., "A.", "B)", "C-") ---
    if re.match(r'^[A-Z][\.\)\-]\s', text):
        return True

    # --- Optional: short all-uppercase line (e.g., "ABSTRACT", "CONCLUSION") ---
    if text.isupper() and len(text) < 60:
        return True

    return False



def ocr_doctr(image_path: str):
    """
    Extract word-level tokens + bounding boxes + header + table flag using docTR and img2table.
    """

    # --- Step 1: Load image properly using PIL ---
    pil_img = PILImage.open(image_path).convert("RGB")
    current_image = PILImage.open(image_path)
    img_width, img_height = pil_img.size

    # --- Step 2: Run Doctr ---
    doc = DocumentFile.from_images(image_path)
    result = model(doc)

    all_words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                size = int(line.geometry[1][1] * img_height) - int(line.geometry[0][1] * img_height)
                line_bbox = [
                    int(line.geometry[0][0] * img_width),
                    int(line.geometry[0][1] * img_height),
                    int(line.geometry[1][0] * img_width),
                    int(line.geometry[1][1] * img_height),
                ]
                line_text = " ".join([word.value for word in line.words])

                for word in line.words:
                    x0 = int(word.geometry[0][0] * img_width)
                    y0 = int(word.geometry[0][1] * img_height)
                    x1 = int(word.geometry[1][0] * img_width)
                    y1 = int(word.geometry[1][1] * img_height)
                    all_words.append({
                        "word": word.value,
                        "bbox": [x0, y0, x1, y1],
                        "height": size,
                        "line_text": line_text,
                        "line_bbox": line_bbox
                    })

    # --- Step 3: Detect tables (img2table) ---
    img2table_img = Img2TableImage(str(image_path), detect_rotation=False)
    img2table_ocr = Img2TableOCR()

    extracted_tables = img2table_img.extract_tables(
        ocr=img2table_ocr,
        implicit_rows=False,
        implicit_columns=False,
        borderless_tables=True,
        min_confidence=50
    )

    table_bboxes = [
        [int(t.bbox.x1), int(t.bbox.y1), int(t.bbox.x2), int(t.bbox.y2)]
        for t in extracted_tables if t.df.shape[0] > 2
    ]

    # --- Step 4: Determine if a word is inside a table ---
    def in_table(bbox, table_bboxes):
        x0, y0, x1, y1 = bbox
        for tx0, ty0, tx1, ty1 in table_bboxes:
            if x0 >= tx0 and y0 >= ty0 and x1 <= tx1 and y1 <= ty1:
                return True
        return False

    for w in all_words:
        w["inside_table"] = in_table(w["bbox"], table_bboxes)
        w["header"] = True if check_headers(w["line_text"], w["line_bbox"], current_image) and len(w["line_text"]) < 80 else False

    return all_words


def normalize_bboxes(tokens: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
    """
    Convert absolute pixel bboxes to LayoutLMv3 expected 0-1000 normalized bboxes.
    Each bbox becomes [x0, y0, x1, y1] in 0-1000.
    """
    w, h = image_size
    norm_tokens = []
    for t in tokens:
        x0, y0, x1, y1 = t['bbox']
        nx0 = int((x0 / w) * 1000)
        ny0 = int((y0 / h) * 1000)
        nx1 = int((x1 / w) * 1000)
        ny1 = int((y1 / h) * 1000)
        # clip
        nx0, ny0, nx1, ny1 = max(0, nx0), max(0, ny0), min(1000, nx1), min(1000, ny1)
        nt = {'word': t['word'], 'bbox': [nx0, ny0, nx1, ny1], 'orig_bbox': t['bbox'], "header": t["header"], "inside_table": t["inside_table"], "line_text": t["line_text"]}
        norm_tokens.append(nt)
    return norm_tokens
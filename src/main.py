from src.parsing.ocr import ocr_pytesseract, normalize_bboxes, ocr_doctr
from src.labelling.synthetic_labelling import synthetic_labeling
from src.tokenizer.tokenizer import tokenize_and_align_labels, sliding_window_chunks
from PIL import Image

MAX_LEN = 512
OVERLAP = 128
STRIDE = MAX_LEN - OVERLAP

def process_image(image_path: str, page_id: int, doc_id: str, ocr="doctr"):
    pil = Image.open(image_path).convert("RGB")
    w, h = pil.size

    # ---- 1. OCR ----
    if ocr=="doctr":
       tokens_raw = ocr_doctr(image_path)
    else:
       tokens_raw = ocr_pytesseract(image_path)
    if not tokens_raw:
        return []

    # ---- 2. Normalize bounding boxes ----
    tokens_norm = normalize_bboxes(tokens_raw, (w, h))
    words = [t['word'] for t in tokens_norm]
    bboxes = [t['bbox'] for t in tokens_norm]

    # ---- 3. Synthetic labels ----
    word_labels = synthetic_labeling(tokens_norm)
   # word_labels = convert_B_to_I(word_labels, LABEL_MAP)

    # ---- 4. Tokenization & alignment ----
    encoding, aligned_labels, token_bboxes = tokenize_and_align_labels(words, bboxes, word_labels)

    # ---- 5. Get token-level words (aligned with input_ids) ----
    word_ids = encoding.word_ids()
    token_words = []
    for idx in word_ids:
        if idx is None:
            token_words.append("[SPECIAL]")  # for [CLS], [SEP], etc.
        else:
            token_words.append(words[idx])

    # ---- 6. Chunking ----
    chunks = sliding_window_chunks(
        encoding, aligned_labels, token_bboxes, max_len=MAX_LEN, stride=STRIDE
    )

    # ---- 7. Build final output entries ----
    results = []
    for i, ch in enumerate(chunks):
        # Determine token range in this chunk
        token_slice = slice(ch["start"], ch["end"])
        chunk_words = token_words[token_slice]

        entry = {
            "id": f"{doc_id}_page{page_id}_chunk{i}",
            "image_path": str(image_path),
            "page": page_id,
            "words": chunk_words,                 # ✅ Added words for readability/debugging
            "input_ids": ch["input_ids"],
            "attention_mask": ch["attention_mask"],
            "labels": ch["labels"],
            "bboxes": ch["bboxes"]
        }
        results.append(entry)

    return results

# ------------- tokenization + alignment --------------
from typing import List
from transformers import LayoutLMv3Processor

MAX_LEN = 512
OVERLAP = 128
STRIDE = MAX_LEN - OVERLAP
MODEL_NAME = "microsoft/layoutlmv3-base"  # used only for tokenizer/processor

processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(words: List[str], bboxes: List[List[int]], word_labels: List[int]):
    """
    Tokenize using processor.tokenizer with is_split_into_words=True,
    then align labels to tokenized outputs using encoding.word_ids().
    Returns:
      encoding (BatchEncoding),
      aligned_labels (list of int per token),
      token_bboxes (list of bbox per token aligned to word)
    """
    tokenizer = processor.tokenizer
    encoding = tokenizer(words,
                         boxes=bboxes, # Added boxes argument
                         return_attention_mask=True,
                         truncation=False)  # we will chunk later

    word_ids = encoding.word_ids()  # list of word index per token
    aligned_labels = []
    token_bboxes = []
    for idx in word_ids:
        if idx is None:
            # special tokens like [CLS], [SEP]
            aligned_labels.append(-100)  # -100 will be ignored by loss functions
            token_bboxes.append([0,0,0,0])
        else:
            aligned_labels.append(word_labels[idx])
            token_bboxes.append(bboxes[idx])
    return encoding, aligned_labels, token_bboxes

# ------------- chunking -------------
def sliding_window_chunks(encoding, aligned_labels, token_bboxes, max_len=MAX_LEN, stride=STRIDE):
    """
    Split encoding into chunks of max_len with stride coverage.
    encoding has input_ids list.
    Return list of dicts with input_ids, attention_mask, labels, bboxes, word_ids slice-aware.
    """
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    n = len(input_ids)
    chunks = []
    start = 0
    while start < n:
        end = min(start + max_len, n)
        chunk_input_ids = input_ids[start:end]
        chunk_attention = attention_mask[start:end]
        chunk_labels = aligned_labels[start:end]
        chunk_bboxes = token_bboxes[start:end]
        chunks.append({
            "input_ids": chunk_input_ids,
            "attention_mask": chunk_attention,
            "labels": chunk_labels,
            "bboxes": chunk_bboxes,
            "start": start,
            "end": end
        })
        if end == n:
            break
        start += stride
    return chunks
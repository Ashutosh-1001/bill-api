# main.py
import io
import os
import re
import tempfile
from decimal import Decimal, InvalidOperation, getcontext
from typing import List, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import requests
from pdf2image import convert_from_bytes
import cv2
from rapidfuzz import fuzz

# Set decimal precision
getcontext().prec = 12

app = FastAPI(title="Bill Extractor (improved)")

# initialize PaddleOCR once
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# regex to find numbers (handles commas and decimals)
NUM_RE = re.compile(r"[-+]?\d{1,3}(?:[,\d]{0,})?(?:\.\d+)?")

TOTAL_KEYWORDS = [
    "grand total", "net payable", "amount payable", "amount due", "total payable",
    "total amount", "net amount", "invoice total", "total"
]

# ---------------------------
# Image helpers (preprocess)
# ---------------------------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def preprocess_pil(pil_img: Image.Image) -> Image.Image:
    # convert to grayscale and increase contrast
    img = pil_img.convert("L")
    # CLAHE style with OpenCV
    arr = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    arr = clahe.apply(arr)
    # denoise (median)
    arr = cv2.medianBlur(arr, 3)
    # optional bilateral filter
    arr = cv2.bilateralFilter(arr, 5, 75, 75)
    # convert back to PIL, sharpen slightly
    pil = Image.fromarray(arr).convert("RGB").filter(ImageFilter.SHARPEN)
    # trim margins (helps some scanned pages)
    pil = ImageOps.expand(pil, border=2)  # small pad
    return pil

# ---------------------------
# PDF -> images
# ---------------------------
def pdf_bytes_to_pil_images(pdf_bytes: bytes) -> List[Image.Image]:
    # convert each PDF page to a PIL image (uses poppler under the hood)
    pil_pages = convert_from_bytes(pdf_bytes, dpi=300)
    return pil_pages

# ---------------------------
# OCR and layout functions
# ---------------------------
def run_ocr_on_pil(pil_img: Image.Image):
    # PaddleOCR expects path or numpy array
    arr = np.array(pil_img)
    res = ocr.ocr(arr)
    return res

def tokens_from_ocr_result(ocr_res) -> List[dict]:
    # flatten results to tokens with bbox + text + confidence + centroid
    tokens = []
    for line in ocr_res:
        for box, (text, conf) in line:
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            tokens.append({
                "text": text.strip(),
                "conf": float(conf),
                "bbox": (x1, y1, x2, y2),
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0
            })
    return tokens

def cluster_tokens_to_rows(tokens: List[dict], y_thresh: int = 12) -> List[List[dict]]:
    if not tokens:
        return []
    tokens_sorted = sorted(tokens, key=lambda t: t['cy'])
    rows = []
    current = [tokens_sorted[0]]
    for tok in tokens_sorted[1:]:
        if abs(tok['cy'] - current[-1]['cy']) <= y_thresh:
            current.append(tok)
        else:
            # sort row left-to-right
            rows.append(sorted(current, key=lambda t: t['cx']))
            current = [tok]
    rows.append(sorted(current, key=lambda t: t['cx']))
    return rows

# ---------------------------
# amount extraction & parsing
# ---------------------------
def find_rightmost_amount_in_row(row_tokens: List[dict]) -> Tuple[Decimal, dict]:
    # scan tokens from right to left for numeric token
    for t in reversed(row_tokens):
        text = t['text'].replace('₹', '').replace('$', '').replace('Rs.', '').strip()
        m = NUM_RE.search(text)
        if m:
            raw = m.group(0).replace(',', '')
            try:
                d = Decimal(raw)
                return d, t
            except InvalidOperation:
                continue
    return None, None

# ---------------------------
# item formation & merging
# ---------------------------
def row_to_candidate_item(row_tokens: List[dict]):
    amount, amount_token = find_rightmost_amount_in_row(row_tokens)
    if amount is None:
        return None
    # name: tokens before amount_token
    name_parts = []
    for t in row_tokens:
        if t is amount_token:
            break
        # skip tiny/confidence=0 tokens
        if t['conf'] < 0.2:
            continue
        name_parts.append(t['text'])
    name = " ".join(name_parts).strip()
    if not name:
        # fallback: use entire row text excluding numeric token
        name = " ".join([t['text'] for t in row_tokens if t is not amount_token]).strip()
    bbox = (
        min([t['bbox'][0] for t in row_tokens]),
        min([t['bbox'][1] for t in row_tokens]),
        max([t['bbox'][2] for t in row_tokens]),
        max([t['bbox'][3] for t in row_tokens]),
    )
    return {
        "item_name": name if name else "UNKNOWN",
        "item_amount": float(amount),
        "item_rate": None,
        "item_quantity": None,
        "raw_conf": min([t['conf'] for t in row_tokens]) if row_tokens else 0.0,
        "bbox": bbox
    }

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def dedupe_and_merge_items(items: List[dict]) -> List[dict]:
    merged = []
    for it in items:
        found = False
        for m in merged:
            name_score = fuzz.token_sort_ratio(it['item_name'], m['item_name'])
            iou = bbox_iou(it['bbox'], m['bbox'])
            amt_close = abs(Decimal(str(it['item_amount'])) - Decimal(str(m['item_amount']))) <= Decimal('0.5')
            # merge rules: high name similarity and some bbox overlap OR amounts very close
            if (name_score > 86 and iou > 0.15) or (amt_close and name_score > 70):
                # choose longer (more descriptive) name
                if len(it['item_name']) > len(m['item_name']):
                    m['item_name'] = it['item_name']
                # average amounts (keeps decimals steady)
                m_amt = Decimal(str(m['item_amount'])); it_amt = Decimal(str(it['item_amount']))
                merged_amt = (m_amt + it_amt) / 2
                m['item_amount'] = float(merged_amt)
                # expand bbox
                m['bbox'] = (
                    min(m['bbox'][0], it['bbox'][0]),
                    min(m['bbox'][1], it['bbox'][1]),
                    max(m['bbox'][2], it['bbox'][2]),
                    max(m['bbox'][3], it['bbox'][3]),
                )
                found = True
                break
        if not found:
            merged.append(it.copy())
    return merged

# ---------------------------
# subtotal/total detection
# ---------------------------
def detect_totals(rows: List[List[dict]]):
    found = {}
    for row in rows:
        text = " ".join([t['text'].lower() for t in row])
        for kw in TOTAL_KEYWORDS:
            if kw in text:
                amt, _ = find_rightmost_amount_in_row(row)
                if amt is not None:
                    found[kw] = float(amt)
    return found

def reconcile_items_with_total(items: List[dict], totals_map: dict):
    sum_items = sum(Decimal(str(it['item_amount'])) for it in items)
    invoice_total = None
    # pick most specific total keyword if present
    for pref in ["grand total","amount payable","amount due","net payable","total"]:
        if pref in totals_map:
            invoice_total = Decimal(str(totals_map[pref]))
            break
    if invoice_total is None and totals_map:
        # fallback to any detected total
        invoice_total = Decimal(str(list(totals_map.values())[0]))
    diff = (invoice_total - sum_items) if invoice_total is not None else None
    return {
        "sum_items": float(sum_items),
        "invoice_total": float(invoice_total) if invoice_total is not None else None,
        "diff": float(diff) if diff is not None else None
    }

# ---------------------------
# Main processing
# ---------------------------
@app.post("/extract-bill-data")
async def extract_bill_data(req: Request):
    body = await req.json()
    url = body.get("document")
    if not url:
        return JSONResponse({"is_success": False, "message": "Please provide 'document' (public URL)."}, status_code=400)

    # download file
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        content = r.content
    except Exception as e:
        return JSONResponse({"is_success": False, "message": f"Could not download document: {e}"}, status_code=400)

    # detect if pdf
    images = []
    if url.lower().endswith(".pdf") or (content[:4] == b"%PDF"):
        try:
            pil_pages = pdf_bytes_to_pil_images(content)
            images = [preprocess_pil(p) for p in pil_pages]
        except Exception as e:
            # fallback: try to open as image
            try:
                img = Image.open(io.BytesIO(content)).convert("RGB")
                images = [preprocess_pil(img)]
            except Exception as e2:
                return JSONResponse({"is_success": False, "message": f"PDF->image conversion failed: {e} / {e2}"}, status_code=400)
    else:
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            images = [preprocess_pil(img)]
        except Exception as e:
            return JSONResponse({"is_success": False, "message": f"Image open failed: {e}"}, status_code=400)

    pagewise_line_items = []
    global_items = []

    for p_idx, pil_img in enumerate(images, start=1):
        ocr_res = run_ocr_on_pil(pil_img)
        tokens = tokens_from_ocr_result(ocr_res)
        rows = cluster_tokens_to_rows(tokens, y_thresh=max(10, int(pil_img.size[1] * 0.003)))  # scale threshold by page height

        # build candidate items from rows
        candidates = []
        for row in rows:
            cand = row_to_candidate_item(row)
            if cand:
                candidates.append(cand)

        # attempt to merge adjacent rows into one item if a row has no amount but next row has small amount and x-start aligns
        merged_candidates = []
        i = 0
        while i < len(rows):
            cand = row_to_candidate_item(rows[i])
            if cand:
                merged_candidates.append(cand)
                i += 1
            else:
                # check next row for amount and combine names if next row's amount exists
                if i + 1 < len(rows):
                    cand_next = row_to_candidate_item(rows[i+1])
                    if cand_next:
                        # join text from rows[i] and rows[i+1] before amount
                        prefix = " ".join([t['text'] for t in rows[i] if t['conf'] > 0.2])
                        cand_next['item_name'] = (prefix + " " + cand_next['item_name']).strip()
                        # recompute bbox
                        cand_next['bbox'] = (
                            min(cand_next['bbox'][0], min([t['bbox'][0] for t in rows[i]])),
                            min(cand_next['bbox'][1], min([t['bbox'][1] for t in rows[i]])),
                            max(cand_next['bbox'][2], max([t['bbox'][2] for t in rows[i]])),
                            max(cand_next['bbox'][3], max([t['bbox'][3] for t in rows[i]])),
                        )
                        merged_candidates.append(cand_next)
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1

        # dedupe page candidates
        page_items = dedupe_and_merge_items(merged_candidates)

        # add to outputs
        bill_items_out = []
        for it in page_items:
            bill_items_out.append({
                "item_name": it['item_name'],
                "item_amount": float(Decimal(str(it['item_amount']))),
                "item_rate": it.get('item_rate'),
                "item_quantity": it.get('item_quantity')
            })
            global_items.append(it)

        pagewise_line_items.append({
            "page_no": str(p_idx),
            "page_type": "Bill Detail",
            "bill_items": bill_items_out
        })

    # final dedupe across pages
    final_items = dedupe_and_merge_items(global_items)
    # detect totals from rows of all pages
    totals_map = {}
    for pil_img in images:
        ocr_res = run_ocr_on_pil(pil_img)
        tokens = tokens_from_ocr_result(ocr_res)
        rows = cluster_tokens_to_rows(tokens, y_thresh=max(10, int(pil_img.size[1] * 0.003)))
        totals_map.update(detect_totals(rows))

    recon = reconcile_items_with_total(final_items, totals_map)

    # build final response list (pagewise already) but also set total_item_count
    total_item_count = len(final_items)

    response = {
        "is_success": True,
        "token_usage": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        },
        "data": {
            "pagewise_line_items": pagewise_line_items,
            "total_item_count": total_item_count,
            "reconciliation": recon,
            "detected_totals": totals_map
        }
    }
    return JSONResponse(response)

from fastapi import FastAPI
from paddleocr import PaddleOCR
import requests, tempfile, re
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_amount(text):
    numbers = re.findall(r"\d+\.\d+|\d+", text.replace(",", ""))
    if numbers:
        try:
            return float(numbers[-1])
        except:
            return None
    return None

def preprocess_image(img: Image.Image):
    # Convert to grayscale and improve contrast for better OCR
    img = img.convert("L")
    img = img.point(lambda x: 0 if x < 140 else 255)
    return img.convert("RGB")

def process_page(pil_img):
    pil_img = preprocess_image(pil_img)
    result = ocr.ocr(np.array(pil_img))
    items = []
    seen = set()

    for line in result:
        for box, (text, conf) in line:
            amount = extract_amount(text)
            if amount is None:
                continue

            # Clean name
            name = text.replace(str(amount), "").strip()

            # Avoid double counting
            key = (name.lower(), amount)
            if key in seen:
                continue
            seen.add(key)

            items.append({
                "item_name": name if name else "UNKNOWN",
                "item_amount": amount,
                "item_rate": amount,
                "item_quantity": 1.0
            })
    return items

@app.get("/")
def check():
    return {"status": "alive"}

@app.post("/extract-bill-data")
def extract_bill(payload: dict):
    url = payload.get("document")
    if not url:
        return {"is_success": False, "message": "Document missing"}

    try:
        file_bytes = requests.get(url, timeout=30).content
    except:
        return {"is_success": False, "message": "Could not download file"}

    pages = []

    # Check if file is PDF
    if file_bytes[:4] == b"%PDF":
        try:
            pages = convert_from_bytes(file_bytes)
        except:
            return {"is_success": False, "message": "PDF error"}
    else:
        try:
            img = Image.open(tempfile.NamedTemporaryFile().name)
        except:
            img = Image.open(tempfile.NamedTemporaryFile().name)
        img = Image.open(tempfile.NamedTemporaryFile().name)

        # Simpler: read via PIL directly
        img = Image.open(tempfile.NamedTemporaryFile().name)

        try:
            img = Image.open(tempfile.NamedTemporaryFile().name)
        except:
            pass

        # Actual load
        try:
            img = Image.open(requests.get(url, stream=True).raw)
        except:
            return {"is_success": False, "message": "Image load failed"}

        pages = [img]

    pagewise = []
    total_items = 0

    for i, p in enumerate(pages):
        items = process_page(p)
        pagewise.append({
            "page_no": str(i + 1),
            "page_type": "Bill Detail",
            "bill_items": items
        })
        total_items += len(items)

    return {
        "is_success": True,
        "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
        "data": {
            "pagewise_line_items": pagewise,
            "total_item_count": total_items
        }
    }

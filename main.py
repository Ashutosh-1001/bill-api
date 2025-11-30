from fastapi import FastAPI
from paddleocr import PaddleOCR
import requests
import tempfile
import re

app = FastAPI()

# load OCR model once
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.get("/")
def home():
    return {"status": "alive"}

def extract_amount(text):
    numbers = re.findall(r"\d+\.\d+|\d+", text)
    if numbers:
        return float(numbers[-1])
    return None

@app.post("/extract-bill-data")
def extract_bill_data(payload: dict):
    url = payload.get("document")
    if not url:
        return {"is_success": False, "message": "No document provided"}

    # download file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(requests.get(url).content)
    tmp.close()

    # OCR
    result = ocr.ocr(tmp.name)

    items = []
    seen = set()

    for line in result:
        for box, (text, conf) in line:
            amount = extract_amount(text)
            if amount:
                name = text.replace(str(amount), "").strip()
                if (name, amount) in seen:
                    continue
                seen.add((name, amount))
                items.append({
                    "item_name": name,
                    "item_amount": amount,
                    "item_rate": amount,
                    "item_quantity": 1.0
                })

    return {
        "is_success": True,
        "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
        "data": {
            "pagewise_line_items": [
                {
                    "page_no": "1",
                    "page_type": "Bill Detail",
                    "bill_items": items
                }
            ],
            "total_item_count": len(items)
        }
    }

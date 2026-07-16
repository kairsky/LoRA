"""Synthetic invoice generator: receipt-like images + ground-truth JSON.

Why synthetic? It proves the registry scales beyond CORD with ZERO external
downloads: the whole pipeline (chat template, masking, training, eval) runs on
locally generated data, works offline, and is fully testable on CPU. It is also
a controllable difficulty dial - more items, noisier layout, new fields.

Each sample is ``{"image": PIL.Image, "target_json": str}`` - the exact
contract the collator expects (same as the CORD builder).
"""

from __future__ import annotations

import json
import random

from PIL import Image, ImageDraw, ImageFont

_VENDORS = (
    "GREEN MARKET", "CITY DELI", "SUNRISE CAFE", "NORTH BAKERY", "STAR GROCERY",
    "RIVER BISTRO", "GOLDEN NOODLES", "URBAN COFFEE", "FRESH CORNER", "OLD MILL PUB",
)
_PRODUCTS = (
    "espresso", "latte", "green tea", "croissant", "bagel", "cheesecake",
    "orange juice", "salad bowl", "tomato soup", "club sandwich", "pasta",
    "pizza slice", "burger", "fries", "spring rolls", "ice cream", "brownie",
    "mineral water", "lemonade", "hot chocolate",
)

IMAGE_SIZE = (420, 560)


def _money(value: float) -> str:
    return f"{value:.2f}"


def generate_invoice(seed: int) -> dict:
    """Deterministically generate one receipt image + its target JSON."""
    rng = random.Random(seed)

    vendor = rng.choice(_VENDORS)
    date = (
        f"{rng.randint(2024, 2026):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    )
    items = []
    for name in rng.sample(_PRODUCTS, k=rng.randint(2, 6)):
        count = rng.randint(1, 4)
        price = rng.randint(150, 2500) / 100.0
        items.append({"name": name, "count": str(count), "price": _money(count * price)})

    subtotal = sum(float(i["price"]) for i in items)
    tax = round(subtotal * rng.choice((0.05, 0.08, 0.10)), 2)
    total = round(subtotal + tax, 2)

    target = {
        "vendor": vendor,
        "date": date,
        "items": items,
        "subtotal": _money(subtotal),
        "tax": _money(tax),
        "total": _money(total),
    }
    image = _render(rng, target)
    return {
        "image": image,
        "target_json": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
    }


def _render(rng: random.Random, target: dict) -> Image.Image:
    """Draw a plausible receipt. Layout jitter keeps it from being trivially
    memorizable; the default PIL bitmap font keeps it dependency-free."""
    img = Image.new("RGB", IMAGE_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)

    x0 = rng.randint(18, 40)
    y = rng.randint(16, 36)

    def line(text: str, f=None, dy: int = 22, x: int | None = None) -> None:
        nonlocal y
        draw.text((x if x is not None else x0, y), text, fill=(20, 20, 20), font=f or small)
        y += dy

    line(target["vendor"], f=font, dy=28)
    line(f"DATE: {target['date']}")
    line("-" * 34)
    for item in target["items"]:
        line(f"{item['name'][:18]:<18} x{item['count']}")
        line(f"{item['price']:>28}", dy=20, x=x0 + rng.randint(0, 8))
    line("-" * 34)
    line(f"SUBTOTAL{target['subtotal']:>20}")
    line(f"TAX{target['tax']:>25}")
    line(f"TOTAL{target['total']:>23}", f=font, dy=28)
    line("THANK YOU!", x=x0 + rng.randint(0, 60))
    return img

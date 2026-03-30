import json
import time
import pathlib
import requests
import ijson

BULK_API = "https://api.scryfall.com/bulk-data"

# Choose ONE:
# - "unique_artwork": one card object per unique artwork
# - "default_cards": one default printing per card
# - "all_cards": every printing (huge)
BULK_TYPE = "unique_artwork"

OUT_DIR = pathlib.Path("data")
IMAGES_DIR = OUT_DIR / "images"
INDEX_PATH = OUT_DIR / "index.jsonl"
BULK_PATH = OUT_DIR / "bulk" / f"{BULK_TYPE}.json"

MAX_IMAGES = None  # set to a small amount for testing first
SLEEP_BETWEEN_IMAGE_DOWNLOADS_S = 0.02  # be polite!!!


def ensure_dirs():
    (OUT_DIR / "bulk").mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_bulk_download_uri(bulk_type: str) -> str:
    r = requests.get(BULK_API, timeout=60)
    r.raise_for_status()
    payload = r.json()

    for item in payload.get("data", []):
        if item.get("type") == bulk_type:
            return item["download_uri"]

    available = [x.get("type") for x in payload.get("data", [])]
    raise RuntimeError(f"Bulk type '{bulk_type}' not found. Available: {available}")


def download_bulk_json(download_uri: str, out_path: pathlib.Path):
    # Stream download to disk so we don't keep it in memory
    print(f"Downloading bulk JSON: {download_uri}")
    with requests.get(download_uri, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Saved bulk file to: {out_path}")


def extract_creature_types(type_line: str) -> list[str]:
    # Example: "Creature — Demon Warlock"
    # Wanted: ["Demon", "Warlock"]
    if "—" not in type_line:
        return []
    rhs = type_line.split("—", 1)[1].strip()
    # Split on spaces, but keep weird/Un- set types as tokens too.
    # All creature subtypes are single tokens
    types = [t.strip() for t in rhs.split() if t.strip()]
    return types


def get_art_url(card: dict) -> str | None:
    # Normal cards
    uris = card.get("image_uris") or {}
    if uris.get("art_crop"):
        return uris["art_crop"]

    # Multi-face cards
    for face in card.get("card_faces") or []:
        furis = face.get("image_uris") or {}
        if furis.get("art_crop"):
            return furis["art_crop"]

    return None


def is_creature_card(card: dict) -> bool:
    type_line = card.get("type_line") or ""
    return "Creature" in type_line


def safe_write_jsonl(fp, obj: dict):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def download_image(url: str, out_path: pathlib.Path):
    if out_path.exists():
        return
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def main():
    ensure_dirs()

    # 1) Get bulk download_uri and download the bulk JSON once
    if not BULK_PATH.exists():
        download_uri = get_bulk_download_uri(BULK_TYPE)
        download_bulk_json(download_uri, BULK_PATH)
    else:
        print(f"Bulk file already exists: {BULK_PATH}")

    # 2) Stream-parse cards, filter creatures, download art, write index.jsonl
    downloaded = 0
    print(f"Parsing bulk file and downloading creature art into: {IMAGES_DIR}")

    seen_ids = set()
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "r", encoding="utf-8") as f_old:
            for line in f_old:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        seen_ids.add(obj["id"])
                except json.JSONDecodeError:
                    pass

    with open(BULK_PATH, "rb") as f_in, open(INDEX_PATH, "a", encoding="utf-8") as f_index:
        # Bulk files are a single big JSON array: [ {card}, {card}, ... ]
        for card in ijson.items(f_in, "item"):
            if not is_creature_card(card):
                continue

            art_url = get_art_url(card)
            if not art_url:
                continue

            card_id = card.get("id")
            if not card_id:
                continue
            if card_id in seen_ids:
                continue

            img_path = IMAGES_DIR / f"{card_id}.jpg"

            try:
                download_image(art_url, img_path)
            except Exception as e:
                print(f"Skip (download failed) {card.get('name')} : {e}")
                continue

            type_line = card.get("type_line") or ""
            entry = {
                "id": card_id,
                "oracle_id": card.get("oracle_id"),
                "name": card.get("name"),
                "set": card.get("set"),
                "collector_number": card.get("collector_number"),
                "type_line": type_line,
                "creature_types": extract_creature_types(type_line),
                "image_path": str(img_path.as_posix()),
                "art_crop_url": art_url,
            }
            safe_write_jsonl(f_index, entry)

            seen_ids.add(card_id)

            downloaded += 1
            if downloaded % 100 == 0:
                print(f"Downloaded {downloaded} creature images...")

            if SLEEP_BETWEEN_IMAGE_DOWNLOADS_S:
                time.sleep(SLEEP_BETWEEN_IMAGE_DOWNLOADS_S)

            if MAX_IMAGES is not None and downloaded >= MAX_IMAGES:
                break

    print(f"Done. Downloaded {downloaded} images.")
    print(f"Index written to: {INDEX_PATH}")


if __name__ == "__main__":
    main()
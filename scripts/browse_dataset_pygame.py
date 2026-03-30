import json
import random
import sys
from pathlib import Path

import pygame
from PIL import Image


INDEX_PATH = Path("data/index.jsonl")
WINDOW_SIZE = (900, 650)  # window size in pixels
IMAGE_AREA_HEIGHT = 540   # top area reserved for the image


def load_index_entries(index_path: Path) -> list[dict]:
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    entries = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines gracefully
                continue

    if not entries:
        raise RuntimeError(f"No valid entries found in {index_path}")

    return entries


def pil_image_to_surface(img: Image.Image) -> pygame.Surface:
    img = img.convert("RGB")
    data = img.tobytes()
    return pygame.image.fromstring(data, img.size, "RGB")


def fit_image_to_rect(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    # Preserve aspect ratio, fit inside max_w x max_h
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    words = text.split()
    lines = []
    current = ""
    for w in words:
        test = (current + " " + w).strip()
        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def draw_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    entry: dict,
    window_w: int,
    window_h: int,
):
    # Bottom panel
    panel_h = window_h - IMAGE_AREA_HEIGHT
    panel_rect = pygame.Rect(0, IMAGE_AREA_HEIGHT, window_w, panel_h)
    pygame.draw.rect(screen, (15, 15, 15), panel_rect)

    name = entry.get("name", "Unknown")
    type_line = entry.get("type_line", "")
    creature_types = entry.get("creature_types") or []

    types_text = ", ".join(creature_types) if creature_types else "(no creature types parsed)"

    # Header
    header = f"{name}"
    header_surf = font.render(header, True, (235, 235, 235))
    screen.blit(header_surf, (16, IMAGE_AREA_HEIGHT + 10))

    # Type line (wrap)
    type_label = f"Type line: {type_line}"
    wrapped = wrap_text(type_label, small_font, window_w - 32)

    y = IMAGE_AREA_HEIGHT + 52
    for line in wrapped[:3]:  # keep it tidy
        surf = small_font.render(line, True, (200, 200, 200))
        screen.blit(surf, (16, y))
        y += 22

    # Creature types line
    ct_label = f"Creature types: {types_text}"
    wrapped_ct = wrap_text(ct_label, small_font, window_w - 32)

    y += 6
    for line in wrapped_ct[:3]:
        surf = small_font.render(line, True, (200, 200, 200))
        screen.blit(surf, (16, y))
        y += 22

    # Controls hint
    hint = "SPACE = next random   |   ESC = quit"
    hint_surf = small_font.render(hint, True, (160, 160, 160))
    hint_x = window_w - hint_surf.get_width() - 16
    hint_y = window_h - 26
    screen.blit(hint_surf, (hint_x, hint_y))

def load_entry_image(entry: dict) -> Image.Image:
    img_path = entry.get("image_path")
    if not img_path:
        raise FileNotFoundError("Entry has no image_path")

    p = Path(img_path)
    if not p.exists():
        # Try relative to project root if paths were saved as posix
        p = Path(*img_path.split("/"))
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    return Image.open(p)


def main():
    entries = load_index_entries(INDEX_PATH)
    rng = random.Random()

    pygame.init()
    try:
        screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("MTG Creature Dataset Browser (SPACE next, ESC quit)")
        clock = pygame.time.Clock()

        font = pygame.font.SysFont(None, 28)
        small_font = pygame.font.SysFont(None, 20)

        current_entry = None
        current_surface = None

        def pick_new():
            nonlocal current_entry, current_surface
            # Keep trying until we load a valid image
            for _ in range(50):
                e = rng.choice(entries)
                try:
                    img = load_entry_image(e)
                    img = fit_image_to_rect(img, WINDOW_SIZE[0], IMAGE_AREA_HEIGHT)
                    current_surface = pil_image_to_surface(img)
                    current_entry = e
                    return
                except Exception:
                    continue
            raise RuntimeError("Could not load any image after multiple attempts.")

        pick_new()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        pick_new()

            # Background
            screen.fill((0, 0, 0))

            # Image (centered in top area)
            if current_surface is not None:
                iw, ih = current_surface.get_size()
                x = (WINDOW_SIZE[0] - iw) // 2
                y = (IMAGE_AREA_HEIGHT - ih) // 2
                screen.blit(current_surface, (x, y))

            # Text overlay
            if current_entry is not None:
                draw_overlay(screen, font, small_font, current_entry, WINDOW_SIZE[0], WINDOW_SIZE[1])

            pygame.display.flip()
            clock.tick(60)

    finally:
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

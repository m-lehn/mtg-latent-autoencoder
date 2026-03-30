import io
import sys
import requests
import pygame
from PIL import Image

SCRYFALL_RANDOM = "https://api.scryfall.com/cards/random"


def get_creature_art_url() -> tuple[str, str]:
    """
    Returns (card_name, art_url) for a random creature card.
    Handles normal cards + multi-face cards.
    """
    params = {"q": "type:creature game:paper -is:digital"}
    r = requests.get(SCRYFALL_RANDOM, params=params, timeout=30)
    r.raise_for_status()
    card = r.json()

    name = card.get("name", "Unknown")

    # Normal cards
    if "image_uris" in card and card["image_uris"]:
        url = card["image_uris"].get("art_crop") or card["image_uris"].get("png")
        if url:
            return name, url

    # Double-faced / special layouts
    faces = card.get("card_faces") or []
    for face in faces:
        uris = face.get("image_uris") or {}
        url = uris.get("art_crop") or uris.get("png")
        if url:
            face_name = face.get("name", name)
            return face_name, url

    raise RuntimeError("No image URL found for this card.")


def download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def pil_to_surface(img: Image.Image) -> pygame.Surface:
    # Convert PIL image to a pygame Surface
    mode = img.mode
    size = img.size
    data = img.tobytes()
    return pygame.image.fromstring(data, size, mode)


def main():
    print("Fetching a random creature from Scryfall...")
    name, art_url = get_creature_art_url()
    print(f"Card: {name}")
    print(f"Art URL: {art_url}")

    img = download_image(art_url)

    pygame.init()
    try:
        # Window size = image size (art_crop is usually ~745x1040)
        w, h = img.size
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"MTG Creature Art — {name} (Esc to quit)")

        surface = pil_to_surface(img)

        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            screen.blit(surface, (0, 0))
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
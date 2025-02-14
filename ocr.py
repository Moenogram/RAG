from PIL import Image
import pytesseract

def extract_text(image_path):
    """
    Öffnet ein Bild und extrahiert mit Tesseract OCR den Text.
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='deu')
    return text

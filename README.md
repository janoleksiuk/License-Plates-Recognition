# Polish License Plate Recognition

System for detecting and reading Polish license plates using OpenCV and Tesseract OCR.

## Features

- Automatic license plate detection
- Polish voivodeship identification
- Multi-plate support
- Noise reduction (blurr, salt and papper, deflection) and image preprocessing
- Configurable detection parameters

## Installation

### Prerequisites
Install Tesseract OCR:
- **Windows**: Download from [Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Tesseract Path
Update `TESSERACT_PATH` in `PlateRecognitionConfig` class to match your installation.

## Usage

```python
from license_plate_recognizer import LicensePlateRecognizer

recognizer = LicensePlateRecognizer(PATH)
recognizer.recognize()
```

## Example Output
```
=== License Plate Recognition Results ===
Plate 1:
  Text: WA12345
  Voivodeship: mazowieckie
  Center coordinates: (245.5, 180.2)
```

## Supported Polish Voivodeships

| Prefix | Voivodeship | Prefix | Voivodeship |
|--------|-------------|--------|-------------|
| D | dolnośląskie | G | pomorskie |
| L | lubelskie | J | warmińsko-mazurskie |
| K | małopolskie | N,O | opolskie |
| W | mazowieckie | S | śląskie |
| Z | zachodniopomorskie | T | świętokrzyskie |
| P | wielkopolskie | B | podlaskie |
| R | podkarpackie | E | łódzkie |
| C | kujawsko-pomorskie | F | lubuskie |

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Tesseract OCR
- pytesseract

See `requirements.txt` for complete dependencies.

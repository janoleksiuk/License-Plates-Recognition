""" Jan Oleksiuk, 2025"""

import os
import cv2
import numpy as np
import pytesseract
from typing import List, Tuple, Optional
from dataclasses import dataclass

PATH = "photos\car3.jpg"


@dataclass
class PlateCandidate:
    """Represents a license plate candidate with its properties."""
    x: int
    y: int
    width: int
    height: int
    score: float = 0.0
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        return self.height / self.width if self.width > 0 else 0
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


class PlateRecognitionConfig:
    """Configuration class for license plate recognition parameters."""
    
    # Tesseract configuration
    TESSERACT_PATH = " YOUR TESSERACT .EXE PATH "
    TESSERACT_CONFIG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Image processing parameters
    CONTRAST_ALPHA = 2.5
    CONTRAST_BETA = 0
    MEDIAN_BLUR_KERNEL = 3
    MORPHOLOGY_KERNEL_SIZE = (3, 3)
    
    # Plate detection thresholds
    MIN_AREA_RATIO = 0.002  # Minimum area as ratio of image area
    MIN_ASPECT_RATIO = 0.2
    MAX_ASPECT_RATIO = 5.0
    
    # OCR parameters
    PLATE_RESIZE_WIDTH = 750
    PLATE_RESIZE_HEIGHT = 200
    
    # Scoring thresholds
    HORIZONTAL_VARIANCE_THRESHOLD = 1000
    VERTICAL_VARIANCE_THRESHOLD = 500


class PolishVoivodeships:
    """Mapping of Polish license plate prefixes to voivodeships."""
    
    VOIVODESHIP_MAP = {
        'D': 'dolnośląskie',
        'L': 'lubelskie', 
        'K': 'małopolskie',
        'W': 'mazowieckie',
        'Z': 'zachodniopomorskie',
        'P': 'wielkopolskie',
        'R': 'podkarpackie',
        'C': 'kujawsko-pomorskie',
        'E': 'łódzkie',
        'F': 'lubuskie',
        'G': 'pomorskie',
        'J': 'warmińsko-mazurskie',
        'N': 'opolskie',
        'O': 'opolskie',
        'S': 'śląskie',
        'T': 'świętokrzyskie',
        'B': 'podlaskie'
    }
    
    @classmethod
    def get_voivodeship(cls, plate_text: str) -> str:
        """Get voivodeship name from plate text."""
        if not plate_text:
            return "Nieznane"
        
        first_letter = plate_text[0].upper()
        return cls.VOIVODESHIP_MAP.get(first_letter, "Nieznane")


class ImageProcessor:
    """Handles image preprocessing operations."""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load image from file path with error handling."""
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return None
        
        return cv2.imread(image_path)
    
    @staticmethod
    def remove_salt_pepper_noise(image: np.ndarray) -> np.ndarray:
        """Remove salt and pepper noise using median blur."""
        return cv2.medianBlur(image, PlateRecognitionConfig.MEDIAN_BLUR_KERNEL)
    
    @staticmethod
    def binarize_image(image: np.ndarray) -> np.ndarray:
        """Convert image to binary using Otsu's method."""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        return cv2.convertScaleAbs(
            image, 
            alpha=PlateRecognitionConfig.CONTRAST_ALPHA, 
            beta=PlateRecognitionConfig.CONTRAST_BETA
        )
    
    @staticmethod
    def remove_border_artifacts(binary_image: np.ndarray) -> np.ndarray:
        """Remove black border artifacts from binary image."""
        img = binary_image.copy()
        h, w = img.shape
        
        try:
            for _ in range(2):
                # Process left and top borders
                for i in range(h):
                    if img[i, 0] == 255:
                        continue
                    for j in range(w):
                        if img[i, j] == 255:
                            break
                        img[i, j] = 255
                
                for i in range(w):
                    if img[0, i] == 255:
                        continue
                    for j in range(h):
                        if img[j, i] == 255:
                            break
                        img[j, i] = 255
                
                # Rotate and process right and bottom borders
                img = cv2.rotate(img, cv2.ROTATE_180)
                
        except IndexError:
            return binary_image
        
        return img


class PlateDetector:
    """Handles license plate detection and filtering."""
    
    @staticmethod
    def find_plate_candidates(image: np.ndarray) -> List[PlateCandidate]:
        """Find potential license plate regions in the image."""
        # Enhance and binarize
        enhanced = ImageProcessor.enhance_contrast(image)
        binary = ImageProcessor.binarize_image(enhanced)
        
        # Morphological operations
        kernel = np.ones(PlateRecognitionConfig.MORPHOLOGY_KERNEL_SIZE, np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        # Find connected components
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        
        candidates = []
        img_h, img_w = dilated.shape
        min_area = img_h * img_w * PlateRecognitionConfig.MIN_AREA_RATIO
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < min_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            candidate = PlateCandidate(x, y, w, h)
            
            # Filter by aspect ratio
            if (PlateRecognitionConfig.MIN_ASPECT_RATIO <= candidate.aspect_ratio <= PlateRecognitionConfig.MAX_ASPECT_RATIO):
                candidates.append(candidate)
        
        return candidates
    
    @staticmethod
    def score_candidates(candidates: List[PlateCandidate], original_image: np.ndarray) -> List[PlateCandidate]:
        """Score plate candidates based on text-like patterns."""
        for candidate in candidates:
            roi = original_image[
                candidate.y:candidate.y + candidate.height,
                candidate.x:candidate.x + candidate.width
            ]
            
            if roi.size == 0:
                continue
            
            # Convert to grayscale if needed
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            # Calculate variance in projections
            horizontal_proj = np.sum(roi_gray, axis=1)
            vertical_proj = np.sum(roi_gray, axis=0)
            
            h_variance = np.var(horizontal_proj)
            v_variance = np.var(vertical_proj)
            
            # Score based on text-like patterns
            score = 0
            if h_variance > PlateRecognitionConfig.HORIZONTAL_VARIANCE_THRESHOLD:
                score += 2
            if v_variance > PlateRecognitionConfig.VERTICAL_VARIANCE_THRESHOLD:
                score += 1
            
            candidate.score = score
        
        # Sort by score and return top 3
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:3]
    
    @staticmethod
    def remove_overlapping_candidates(candidates: List[PlateCandidate]) -> List[PlateCandidate]:
        """Remove overlapping plate candidates, keeping smaller ones."""
        if len(candidates) <= 1:
            return candidates
        
        candidates_list = list(candidates)
        keep_indices = set(range(len(candidates_list)))
        
        for i in range(len(candidates_list)):
            if i not in keep_indices:
                continue
            
            for j in range(i + 1, len(candidates_list)):
                if j not in keep_indices:
                    continue
                
                if PlateDetector._rectangles_intersect(candidates_list[i], candidates_list[j]):
                    # Keep the smaller area (likely more precise)
                    if candidates_list[i].area > candidates_list[j].area:
                        keep_indices.discard(i)
                        break
                    else:
                        keep_indices.discard(j)
        
        return [candidates_list[i] for i in sorted(keep_indices)]
    
    @staticmethod
    def _rectangles_intersect(rect1: PlateCandidate, rect2: PlateCandidate) -> bool:
        """Check if two rectangles intersect."""
        x_left = max(rect1.x, rect2.x)
        y_top = max(rect1.y, rect2.y)
        x_right = min(rect1.x + rect1.width, rect2.x + rect2.width)
        y_bottom = min(rect1.y + rect1.height, rect2.y + rect2.height)
        
        return x_left < x_right and y_top < y_bottom


class OCRProcessor:
    """Handles OCR operations for license plate text extraction."""
    
    def __init__(self):
        """Initialize OCR processor with Tesseract configuration."""
        pytesseract.pytesseract.tesseract_cmd = PlateRecognitionConfig.TESSERACT_PATH
    
    def extract_text_from_plate(self, plate_image: np.ndarray) -> str:
        """Extract text from a license plate image."""
        # Rotate if height > width
        h, w = plate_image.shape
        if h > w:
            plate_image = cv2.rotate(plate_image, cv2.ROTATE_90_CLOCKWISE)
        
        # Resize for better OCR
        resized = cv2.resize(
            plate_image, 
            (PlateRecognitionConfig.PLATE_RESIZE_WIDTH, PlateRecognitionConfig.PLATE_RESIZE_HEIGHT),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Preprocess for OCR
        binary = ImageProcessor.binarize_image(resized)
        kernel = np.ones(PlateRecognitionConfig.MORPHOLOGY_KERNEL_SIZE, np.uint8)
        
        # Morphological operations
        dilated = cv2.dilate(binary, kernel, iterations=3)
        eroded = cv2.erode(dilated, kernel, iterations=3)
        
        # Remove border artifacts
        cleaned = ImageProcessor.remove_border_artifacts(eroded)
        
        # Extract text using OCR
        try:
            text = pytesseract.image_to_string(cleaned, config=PlateRecognitionConfig.TESSERACT_CONFIG)
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            print("Tesseract not found. Please check the installation path.")
            return ""


class LicensePlateRecognizer:
    """Main class for license plate recognition system."""
    
    def __init__(self, image_path: str):
        """Initialize the recognizer with an image path."""
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
        self.ocr_processor = OCRProcessor()
    
    def preprocess_image(self) -> bool:
        """Load and preprocess the input image."""
        # Load image
        self.original_image = ImageProcessor.load_image(self.image_path)
        if self.original_image is None:
            return False
        
        # Convert to grayscale and denoise
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = ImageProcessor.remove_salt_pepper_noise(gray)
        
        return True
    
    def detect_plates(self) -> List[PlateCandidate]:
        """Detect license plates in the image."""
        if self.processed_image is None:
            raise ValueError("Image not preprocessed. Call preprocess_image() first.")
        
        # Find and filter candidates
        candidates = PlateDetector.find_plate_candidates(self.processed_image)
        scored_candidates = PlateDetector.score_candidates(candidates, self.processed_image)
        final_candidates = PlateDetector.remove_overlapping_candidates(scored_candidates)
        
        return final_candidates
    
    def extract_plate_texts(self, candidates: List[PlateCandidate]) -> List[str]:
        """Extract text from detected plate candidates."""
        plate_texts = []
        
        for candidate in candidates:
            # Extract plate region
            plate_roi = self.processed_image[
                candidate.y:candidate.y + candidate.height,
                candidate.x:candidate.x + candidate.width
            ]
            
            # Extract text
            text = self.ocr_processor.extract_text_from_plate(plate_roi)
            plate_texts.append(text)
        
        return plate_texts
    
    def display_results(self, candidates: List[PlateCandidate], plate_texts: List[str]):
        """Display recognition results."""
        print("\n=== License Plate Recognition Results ===")
        
        for i, (candidate, text) in enumerate(zip(candidates, plate_texts)):
            voivodeship = PolishVoivodeships.get_voivodeship(text)
            center_x, center_y = candidate.center
            
            print(f"Plate {i+1}:")
            print(f"  Text: {text if text else 'Nieznane'}")
            print(f"  Voivodeship: {voivodeship}")
            print(f"  Center coordinates: ({center_x:.1f}, {center_y:.1f})")
            print()
        
        # Draw rectangles on original image
        display_image = self.original_image.copy()
        for candidate in candidates:
            cv2.rectangle(display_image, candidate.as_tuple(), (0, 255, 0), 2)
        
        # Show result
        cv2.imshow('Detected License Plates', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def recognize(self):
        """Main recognition pipeline."""
        print("Starting license plate recognition...")
        
        # Preprocess image
        if not self.preprocess_image():
            print("Failed to load or preprocess image.")
            return
        
        # Detect plates
        candidates = self.detect_plates()
        if not candidates:
            print("No license plates detected.")
            return
        
        print(f"Found {len(candidates)} potential license plate(s).")
        
        # Extract text
        plate_texts = self.extract_plate_texts(candidates)
        
        # Display results
        self.display_results(candidates, plate_texts)


def main():
    """Main function to run the license plate recognition system."""
    # Configuration
    image_path = PATH  # Change this path as needed
    
    # Create and run recognizer
    recognizer = LicensePlateRecognizer(image_path)
    recognizer.recognize()


if __name__ == "__main__":
    main()
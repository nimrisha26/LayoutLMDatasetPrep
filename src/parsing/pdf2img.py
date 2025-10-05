from pdf2image import convert_from_path
from pathlib import Path
from typing import List

def pdf_to_images(pdf_path: str, IMAGES_DIR: str, dpi: int = 200, 
                  poppler_path: str = r"C:\Users\pavit\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin") -> List[Path]:
    """Convert each page of a PDF to an image and return list of saved image paths."""
    
    # Ensure output directory exists
    output_dir = Path(IMAGES_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert PDF to list of PIL images
    images = convert_from_path(
        pdf_path=pdf_path,
        dpi=dpi,
        poppler_path=poppler_path
    )

    saved_paths = []
    base = Path(pdf_path).stem

    # Save each image page
    for i, img in enumerate(images):
        out_path = output_dir / f"{base}_page_{i:03d}.png"
        img.save(out_path)
        saved_paths.append(out_path)
    
    return saved_paths

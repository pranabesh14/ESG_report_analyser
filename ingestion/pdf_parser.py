"""
Document Structure + Layout Understanding.

Uses PyMuPDF (fitz) to walk the PDF page by page, classify blocks as
heading/paragraph/image/table-candidate using font-size heuristics + block
geometry, and emit raw DocUnits. This is intentionally the "dumb but
reliable" layer -- semantic work happens in later stages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pdf_rag_kb.core.schemas import BoundingBox, DocUnit, UnitType, new_id

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    unit_id: str
    doc_id: str
    page_number: int
    image_bytes: bytes
    media_type: str
    bbox: BoundingBox
    nearby_caption: str = ""


class PDFParser:
    """Wraps PyMuPDF. Kept as a thin class so it can be swapped for
    `unstructured.io` or another backend without touching callers.
    """

    # Font size, relative to page median, above which a block is
    # considered a heading rather than body text.
    HEADING_SIZE_RATIO = 1.25

    def __init__(self, doc_id: str):
        self.doc_id = doc_id

    def parse(self, pdf_path: str) -> tuple[list[DocUnit], list[ExtractedImage]]:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        units: list[DocUnit] = []
        images: list[ExtractedImage] = []
        current_heading = ""

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            median_size = self._median_font_size(page)

            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # text block
                    block_text, max_size = self._flatten_text_block(block)
                    if not block_text.strip():
                        continue
                    bbox = BoundingBox(*block["bbox"])
                    is_heading = max_size >= median_size * self.HEADING_SIZE_RATIO and len(block_text) < 150

                    if is_heading:
                        current_heading = block_text.strip()
                        units.append(DocUnit(
                            unit_id=new_id("unit"), doc_id=self.doc_id, page_number=page_number,
                            unit_type=UnitType.HEADING, bbox=bbox, raw_text=block_text.strip(),
                            section_heading=current_heading,
                        ))
                    else:
                        units.append(DocUnit(
                            unit_id=new_id("unit"), doc_id=self.doc_id, page_number=page_number,
                            unit_type=UnitType.PARAGRAPH, bbox=bbox, raw_text=block_text.strip(),
                            section_heading=current_heading,
                        ))

            # Images (charts, photos, diagrams -- vision stage disambiguates which)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception as e:
                    logger.warning("Failed to extract image xref=%s page=%s: %s", xref, page_number, e)
                    continue
                img_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                rects = page.get_image_rects(xref)
                bbox = BoundingBox(*rects[0]) if rects else BoundingBox(0, 0, 0, 0)

                images.append(ExtractedImage(
                    unit_id=new_id("img"), doc_id=self.doc_id, page_number=page_number,
                    image_bytes=img_bytes, media_type=f"image/{ext}", bbox=bbox,
                    nearby_caption=self._find_nearby_caption(page, bbox),
                ))

        doc.close()
        logger.info("Parsed %s: %d text units, %d images", pdf_path, len(units), len(images))
        return units, images

    @staticmethod
    def _median_font_size(page) -> float:
        sizes = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(span["size"])
        if not sizes:
            return 10.0
        sizes.sort()
        return sizes[len(sizes) // 2]

    @staticmethod
    def _flatten_text_block(block) -> tuple[str, float]:
        text_parts = []
        max_size = 0.0
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text_parts.append(span["text"])
                max_size = max(max_size, span["size"])
        return " ".join(text_parts), max_size

    @staticmethod
    def _find_nearby_caption(page, bbox: BoundingBox, vertical_window: float = 40.0) -> str:
        """Heuristic: text block starting just below the image, often a caption."""
        text_dict = page.get_text("dict")
        best_text, best_dy = "", vertical_window
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            bx0, by0, bx1, by1 = block["bbox"]
            dy = by0 - bbox.y1
            if 0 <= dy <= best_dy:
                text, _ = PDFParser._flatten_text_block(block)
                if text.strip():
                    best_text, best_dy = text.strip(), dy
        return best_text

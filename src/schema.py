from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentChunk:
    chunk_id: str
    page_number: int
    content_type: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    
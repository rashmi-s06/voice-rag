import csv
import re
from pathlib import Path
from typing import List
from src.rag.msmarco_loader import MSMarcoPassage

def chunk_text(text: str, max_chunk_len: int = 500) -> List[str]:
    """Splits raw text into semantic chunks, respecting paragraph and sentence boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        
    chunks = []
    for p in paragraphs:
        if len(p) <= max_chunk_len:
            chunks.append(p)
        else:
            # Split large paragraph into sentences
            sentences = re.split(r'(?<=[.!?]) +', p)
            curr = []
            curr_len = 0
            for s in sentences:
                if curr_len + len(s) > max_chunk_len and curr:
                    chunks.append(" ".join(curr))
                    curr = [s]
                    curr_len = len(s)
                else:
                    curr.append(s)
                    curr_len += len(s)
            if curr:
                chunks.append(" ".join(curr))
                
    return [c.strip() for c in chunks if len(c.strip()) > 10]

def parse_uploaded_file(file_path: str) -> List[MSMarcoPassage]:
    """Parses an uploaded dataset file (.tsv, .csv, .pdf, .txt, .md) into indexed passages."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {file_path}")
        
    ext = path.suffix.lower()
    passages: List[MSMarcoPassage] = []
    stem = path.stem.upper()[:8]

    # --- 1. TSV Format (MS MARCO standard) ---
    if ext == ".tsv":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                if not row:
                    continue
                if idx == 0 and ("passage_id" in row[0].lower() or "id" in row[0].lower()):
                    continue
                pid = str(row[0]).strip() if len(row) > 1 else f"{stem}_{idx:03d}"
                text = str(row[1]).strip() if len(row) > 1 else str(row[0]).strip()
                if text:
                    passages.append(MSMarcoPassage(passage_id=pid, text=text, metadata={"source": path.name}))

    # --- 2. CSV Format ---
    elif ext == ".csv":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = []
            text_col_idx = 1
            id_col_idx = 0
            for idx, row in enumerate(reader):
                if not row:
                    continue
                if idx == 0:
                    header = [h.lower().strip() for h in row]
                    for target in ["passage_text", "text", "content", "passage", "body", "answer"]:
                        if target in header:
                            text_col_idx = header.index(target)
                            break
                    for target in ["passage_id", "id", "pid"]:
                        if target in header:
                            id_col_idx = header.index(target)
                            break
                    continue
                    
                pid = str(row[id_col_idx]).strip() if len(row) > id_col_idx else f"{stem}_{idx:03d}"
                text = str(row[text_col_idx]).strip() if len(row) > text_col_idx else " ".join(row)
                if text:
                    passages.append(MSMarcoPassage(passage_id=pid, text=text, metadata={"source": path.name}))

    # --- 3. PDF Format ---
    elif ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        full_text_pages = []
        for p_idx, page in enumerate(reader.pages, 1):
            extracted = page.extract_text() or ""
            if extracted.strip():
                full_text_pages.append((p_idx, extracted.strip()))
                
        chunk_idx = 1
        for p_idx, page_content in full_text_pages:
            chunks = chunk_text(page_content)
            for c in chunks:
                pid = f"{stem}_P{p_idx:02d}_{chunk_idx:03d}"
                passages.append(MSMarcoPassage(passage_id=pid, text=c, metadata={"source": path.name, "page": p_idx}))
                chunk_idx += 1

    # --- 4. Plain Text & Markdown ---
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        chunks = chunk_text(content)
        for idx, c in enumerate(chunks, 1):
            pid = f"{stem}_{idx:03d}"
            passages.append(MSMarcoPassage(passage_id=pid, text=c, metadata={"source": path.name}))

    return passages

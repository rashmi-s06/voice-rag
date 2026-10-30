import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class MSMarcoPassage:
    passage_id: str
    text: str
    metadata: Dict[str, str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        return asdict(self)

SAMPLE_MSMARCO_PASSAGES = [
    (
        "100001",
        "The speed of light in vacuum, commonly denoted c, is a universal physical constant important in many areas of physics. Its exact value is defined as 299,792,458 metres per second (approximately 300,000 km/s, or 186,000 mi/s). It is the upper limit for the speed at which conventional matter or information can travel.",
        {"topic": "Physics", "title": "Speed of light"}
    ),
    (
        "100002",
        "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions. Chemical energy produced by the mitochondria is stored in a small molecule called adenosine triphosphate (ATP). They are often described as the powerhouses of the cell.",
        {"topic": "Biology", "title": "Mitochondria function"}
    ),
    (
        "100003",
        "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. Oxygen is released as a byproduct of this process.",
        {"topic": "Biology", "title": "Photosynthesis process"}
    ),
    (
        "100004",
        "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.",
        {"topic": "Computer Science", "title": "Python Programming"}
    ),
    (
        "100005",
        "A vector database is a specialized database designed to store, manage, and search high-dimensional vector representations of data. Vector databases index vectors for fast similarity search using approximate nearest neighbors algorithms like HNSW and IVFFlat.",
        {"topic": "Artificial Intelligence", "title": "Vector Database Definition"}
    ),
    (
        "100006",
        "The Taj Mahal is an ivory-white marble mausoleum on the south bank of the Yamuna river in the Indian city of Agra. It was commissioned in 1632 by the Mughal emperor Shah Jahan to house the tomb of his favourite wife, Mumtaz Mahal.",
        {"topic": "History", "title": "Taj Mahal History"}
    ),
    (
        "100007",
        "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China–Nepal border runs across its summit point. Its elevation of 8,848.86 m was most recently established in 2020 by the Chinese and Nepali authorities.",
        {"topic": "Geography", "title": "Mount Everest Height"}
    ),
    (
        "100008",
        "Retrieval-Augmented Generation (RAG) is a technique that enhances large language model accuracy by retrieving relevant external facts from an authoritative knowledge base before generating a response. This significantly reduces hallucinations and keeps responses up to date.",
        {"topic": "Artificial Intelligence", "title": "What is RAG"}
    ),
    (
        "100009",
        "The human heart is a muscular organ about the size of a fist, located just behind and slightly left of the breastbone. It pumps blood through the network of arteries and veins called the cardiovascular system, delivering oxygen and nutrients throughout the body.",
        {"topic": "Medicine", "title": "Human Heart Anatomy"}
    ),
    (
        "100010",
        "Insulin is a peptide hormone produced by beta cells of the pancreatic islets; it is considered to be the main anabolic hormone of the body. It regulates the metabolism of carbohydrates, fats and protein by promoting the absorption of glucose from the blood into liver, fat and skeletal muscle cells.",
        {"topic": "Medicine", "title": "Insulin Function"}
    ),
    (
        "100011",
        "The Indian Space Research Organisation (ISRO) successfully launched the Chandrayaan-3 lunar exploration mission on July 14, 2023. The lander Vikram and rover Pragyan made a soft landing near the lunar south pole region on August 23, 2023, making India the fourth country to achieve a soft landing.",
        {"topic": "Space Exploration", "title": "Chandrayaan-3 Mission"}
    ),
    (
        "100012",
        "Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web. It is capable of speech transcription in dozens of languages as well as translation into English.",
        {"topic": "Machine Learning", "title": "Whisper Speech Recognition"}
    ),
]

class MSMarcoLoader:
    """Loads, parses, and provides passages from MS MARCO TSV or pre-packaged collections."""

    @staticmethod
    def ensure_sample_tsv(target_path: Path) -> Path:
        """Creates sample MS MARCO TSV if not already present."""
        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["passage_id", "passage_text", "metadata"])
                for pid, text, meta in SAMPLE_MSMARCO_PASSAGES:
                    writer.writerow([pid, text, json.dumps(meta)])
            logger.info(f"Created default MS MARCO sample dataset at {target_path}")
        return target_path

    @staticmethod
    def load_tsv(file_path: Path) -> List[MSMarcoPassage]:
        """Loads passages from any MS MARCO TSV file.
        Supports standard MS MARCO:
          <passage_id>\t<passage_text>
        or with headers:
          passage_id\tpassage_text\tmetadata
        """
        passages: List[MSMarcoPassage] = []
        if not file_path.exists():
            raise FileNotFoundError(f"MS MARCO dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_no, row in enumerate(reader):
                if not row or len(row) < 2:
                    continue

                # Header detection
                if line_no == 0 and ("passage_id" in row[0].lower() or "id" in row[0].lower()):
                    continue

                pid = str(row[0]).strip()
                text = str(row[1]).strip()
                meta = {}
                if len(row) >= 3 and row[2]:
                    try:
                        meta = json.loads(row[2])
                    except Exception:
                        meta = {"raw_meta": row[2]}

                if pid and text:
                    passages.append(MSMarcoPassage(passage_id=pid, text=text, metadata=meta))

        logger.info(f"Loaded {len(passages)} MS MARCO passages from {file_path}")
        return passages

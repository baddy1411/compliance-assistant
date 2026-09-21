"""Kafka producer that publishes sample compliance documents.

Reads every file in the ``data/`` directory next to this script via
:func:`ingest.load_document` and sends each one as a JSON message::

    {"doc_id": "...", "title": "...", "text": "..."}

to ``settings.kafka_topic`` for the streaming pipeline to consume.

Run with: ``python producer.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import ingest


def main() -> None:
    """Load sample documents from data/ and publish them to Kafka."""
    try:
        from kafka import KafkaProducer
    except ImportError:
        raise SystemExit(
            "The 'kafka-python' package is required for the producer but is "
            "not installed. Install it with: pip install kafka-python"
        )

    from config import get_settings

    settings = get_settings()

    data_dir = Path(__file__).resolve().parent / "data"
    if not data_dir.is_dir():
        raise SystemExit(f"Sample data directory not found: {data_dir}")

    files = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )
    if not files:
        raise SystemExit(f"No sample documents found in {data_dir}")

    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        value_serializer=lambda payload: json.dumps(payload).encode("utf-8"),
    )
    try:
        for path in files:
            try:
                document = ingest.load_document(str(path))
            except (
                FileNotFoundError,
                ValueError,
                ModuleNotFoundError,
                OSError,
                UnicodeDecodeError,
            ) as exc:
                print(f"Skipping {path.name}: could not load ({exc})")
                continue
            payload = {
                "doc_id": document.doc_id,
                "title": document.title,
                "text": document.text,
            }
            record = producer.send(settings.kafka_topic, value=payload).get(
                timeout=30
            )
            print(
                f"Sent '{document.doc_id}' to '{settings.kafka_topic}' "
                f"(partition {record.partition}, offset {record.offset})"
            )
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()

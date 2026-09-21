"""Kafka consumer that streams compliance documents into the vector store.

Consumes JSON messages of the form::

    {"doc_id": "...", "title": "...", "text": "...", "metadata": {...}}

from ``settings.kafka_topic``, indexes each document via
:meth:`rag.ComplianceAssistant.ingest_document`, and commits offsets only
after a message has been handled. Malformed messages are skipped (and
committed, to avoid poison-pill loops) with a warning printed to stdout.

Run with: ``python streaming_pipeline.py``
"""

from __future__ import annotations

import json
from typing import Any


def _parse_payload(value: Any) -> tuple[str, str, str, dict[str, Any]]:
    """Validate a consumed message value into (doc_id, title, text, metadata).

    Raises:
        ValueError: If the payload is missing required fields.
        TypeError: If the payload or its fields have the wrong types.
    """
    if not isinstance(value, dict):
        raise TypeError(
            f"message value must be a JSON object, got {type(value).__name__}"
        )
    doc_id = value.get("doc_id")
    text = value.get("text")
    if not doc_id or not isinstance(doc_id, str):
        raise ValueError("message is missing required string field 'doc_id'")
    if not text or not isinstance(text, str):
        raise ValueError("message is missing required string field 'text'")
    title = value.get("title") or doc_id
    if not isinstance(title, str):
        raise TypeError("field 'title' must be a string")
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise TypeError("field 'metadata' must be a JSON object")
    return doc_id, title, text, metadata


def main() -> None:
    """Consume documents from Kafka and index them until interrupted."""
    try:
        from kafka import KafkaConsumer
    except ImportError:
        raise SystemExit(
            "The 'kafka-python' package is required for the streaming "
            "pipeline but is not installed. "
            "Install it with: pip install kafka-python"
        )

    from config import get_settings
    from rag import ComplianceAssistant

    settings = get_settings()
    assistant = ComplianceAssistant(settings=settings)

    consumer = KafkaConsumer(
        settings.kafka_topic,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="compliance-assistant-indexer",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
    )
    print(
        f"Consuming documents from Kafka topic '{settings.kafka_topic}'. "
        "Press Ctrl+C to stop."
    )
    try:
        for message in consumer:
            try:
                doc_id, title, text, metadata = _parse_payload(message.value)
                chunks = assistant.ingest_document(doc_id, title, text, metadata)
            except (ValueError, RuntimeError) as exc:
                # Skip poison messages but still commit below so the
                # consumer can make progress past them.
                print(f"Skipping message at offset {message.offset}: {exc}")
            else:
                print(
                    f"Indexed '{doc_id}': {chunks} chunk(s) "
                    f"(partition {message.partition}, offset {message.offset})"
                )
            consumer.commit()
    except KeyboardInterrupt:
        print("\nShutdown requested, closing consumer...")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()

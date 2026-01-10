import json
import time
from kafka import KafkaProducer


def stream_documents():
    """Yield sample regulatory and policy documents.

    In a real system you would parse PDF or HTML documents, extract text and
    send them to a vector store.  Here we just send a couple of examples
    sequentially.
    """
    documents = [
        {
            "doc_id": "gdpr",
            "text": "Article 5 of the GDPR specifies principles relating to personal data processing.",
        },
        {
            "doc_id": "internal_policy",
            "text": "Our company privacy policy requires explicit consent for data sharing.",
        },
    ]
    for doc in documents:
        yield doc


def main():
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    topic = "documents"
    for doc in stream_documents():
        producer.send(topic, value=doc)
        print(f"Produced: {doc}")
        time.sleep(1)


if __name__ == "__main__":
    main()
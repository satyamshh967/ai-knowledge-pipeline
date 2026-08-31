from uuid import uuid4

import numpy as np

from app.models import Chunk
from app.vector_store import VectorStore


def test_delete_by_document_id():

    vector_store = VectorStore(
        path="./data/test_chroma"
    )

    document_id = uuid4()

    chunk = Chunk(
        document_id=document_id,
        content="This is a test chunk.",
        position=0
    )

    embedding = np.array([
        [0.1, 0.2, 0.3]
    ])

    vector_store.add_chunks(
        [chunk],
        embedding
    )

    result = vector_store.delete_by_document_id(
        str(document_id)
    )

    assert result == 1

    remaining = vector_store.collection.get(
        where={
            "document_id": str(document_id)
        }
    )

    assert len(remaining["ids"]) == 0
    
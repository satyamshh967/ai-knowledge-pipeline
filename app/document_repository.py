import json
from pathlib import Path
from uuid import UUID

from app.models import Document


class DocumentRepository:

    def __init__(self, path: str = "./data/documents.json"):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not self.path.exists():
            self.path.write_text("[]")

    def _load(self):
        return json.loads(
            self.path.read_text()
        )

    def _save(self, documents):
        self.path.write_text(
            json.dumps(
                documents,
                indent=2
            )
        )

    def add(self, document: Document):
        documents = self._load()

        documents.append(
            document.model_dump(mode="json")
        )

        self._save(documents)

    def get_all(self):
        return [
            Document(**document)
            for document in self._load()
        ]

    def get(self, document_id: UUID):
        for document in self.get_all():
            if document.id == document_id:
                return document

        return None

    def delete(self, document_id: UUID):
        documents = self._load()

        remaining = [
            document
            for document in documents
            if document["id"] != str(document_id)
        ]

        if len(remaining) == len(documents):
            return False

        self._save(remaining)

        return True
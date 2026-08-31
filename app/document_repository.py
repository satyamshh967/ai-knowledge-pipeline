from uuid import UUID

from sqlalchemy import select

from app.database import SessionLocal
from app.database_models import DocumentRecord
from app.models import Document


class DocumentRepository:

    def add(self, document: Document):
        with SessionLocal() as session:
            record = DocumentRecord(
                id=str(document.id),
                title=document.title,
                source=str(document.source),
                content=document.content,
                created_at=document.created_at
            )

            session.add(record)
            session.commit()

    def get_all(self):
        with SessionLocal() as session:
            records = session.scalars(
                select(DocumentRecord).order_by(
                    DocumentRecord.created_at
                )
            ).all()

            return [
                Document(
                    id=UUID(record.id),
                    title=record.title,
                    source=record.source,
                    content=record.content,
                    created_at=record.created_at
                )
                for record in records
            ]

    def get(self, document_id: UUID):
        with SessionLocal() as session:
            record = session.get(
                DocumentRecord,
                str(document_id)
            )

            if record is None:
                return None

            return Document(
                id=UUID(record.id),
                title=record.title,
                source=record.source,
                content=record.content,
                created_at=record.created_at
            )

    def get_by_source(self, source: str):
        with SessionLocal() as session:
            record = session.scalar(
                select(DocumentRecord).where(
                    DocumentRecord.source == source
                )
            )

            if record is None:
                return None

            return Document(
                id=UUID(record.id),
                title=record.title,
                source=record.source,
                content=record.content,
                created_at=record.created_at
            )

    def update(self, document: Document):
        with SessionLocal() as session:
            record = session.get(
                DocumentRecord,
                str(document.id)
            )

            if record is None:
                return False

            record.title = document.title
            record.source = str(document.source)
            record.content = document.content
            record.created_at = document.created_at

            session.commit()

            return True

    def delete(self, document_id: UUID):
        with SessionLocal() as session:
            record = session.get(
                DocumentRecord,
                str(document_id)
            )

            if record is None:
                return False

            session.delete(record)
            session.commit()

            return True
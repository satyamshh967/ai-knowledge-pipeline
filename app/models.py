from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    source: HttpUrl
    content: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    position: int
    metadata: dict = Field(default_factory=dict)
    
    
class RetrievedChunk(BaseModel):
    content: str
    score: float
    document_id: UUID
    position: int
    title: str
    source: str
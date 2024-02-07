from pydantic import BaseModel


class HitsResponse(BaseModel):
    count: int
    hits: list[str]


class PersistedResponse(BaseModel):
    count: int
    filepath: str | None

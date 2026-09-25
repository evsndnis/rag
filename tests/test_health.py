from fastapi.testclient import TestClient

import app.main as main
from app.main import app


def test_health(monkeypatch) -> None:
    # lifespan строит RAG-цепочку (Qdrant + эмбеддер) — в CI их нет, подменяем заглушкой
    monkeypatch.setattr(main, "build_rag_chain", lambda: (None, None))
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

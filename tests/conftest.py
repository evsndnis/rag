import os
from unittest.mock import MagicMock

import pytest

# Без этого Gradio при импорте app.main шлёт телеметрию и проверку версии в фоновых
# потоках; их логи прилетают после закрытия вывода pytest → "Logging error".
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from app import main  # noqa: E402


@pytest.fixture(autouse=True)
def stub_rag_chain(monkeypatch):
    """Не поднимать настоящую цепочку в lifespan: в тестах нет ни Qdrant, ни LLM.

    Тесты, которым нужен конкретный ответ цепочки, перекрывают заглушку своим @patch.
    """
    monkeypatch.setattr(main, "build_rag_chain", lambda: (MagicMock(), MagicMock()))

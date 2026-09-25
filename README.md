# RAG
FastAPI RAG service

Документация и swagger: https://130-49-143-10.nip.io/docs

## Качество (RAG-оценка)

Оценка на golden-датасете из 10 вопросов: 7 на английском по трём разделам документации scikit-learn (`linear_model`, `tree`, `model_evaluation`), 2 на русском для проверки мультиязычного поиска и 1 мета-вопрос про `about.md`.

| Метрика | Значение | Что измеряет |
|---|---|---|
| Retriever Recall@4 | **1.00** (10/10) | в топ-4 чанках есть хотя бы один из нужного раздела документации |
| Faithfulness (RAGAS) | **0.86** | ответ опирается на найденный контекст, без выдумок |
| Answer Relevancy (RAGAS) | **0.94** | ответ по существу вопроса |

Конфигурация прогона:
- LLM: `meta-llama/llama-3.3-70b-instruct` (OpenRouter); та же модель выступает судьёй RAGAS
- эмбеддинги: `intfloat/multilingual-e5-small`
- `top_k = 4`

Recall@4 считается по совпадению ключевого слова в URL источника, то есть на уровне раздела документации, а не конкретного фрагмента.

Воспроизвести: [notebooks/rag_eval.ipynb](notebooks/rag_eval.ipynb). Итоговые значения сохраняются в [notebooks/rag_metrics.json](notebooks/rag_metrics.json).

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

### Эксперимент: reranker

Между retriever'ом и сборкой контекста добавлен cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`: retriever достаёт top-20 чанков, cross-encoder заново оценивает каждую пару «вопрос — чанк», и в контекст LLM попадают 5 лучших.

| Метрика | Без reranker (top-4) | С reranker (top-20 → top-5) | Δ |
|---|---|---|---|
| Retriever Recall | 1.00 | 1.00 | 0 |
| Faithfulness (RAGAS) | 0.86 | **0.90** | +0.04 |
| Answer Relevancy (RAGAS) | 0.94 | 0.94 | 0 |

Что это значит:
- **Faithfulness выросла на 0.04**: с переоценённым контекстом ответы чуть точнее опираются на источники.
- **Answer Relevancy и Recall не изменились.** Recall и без reranker'а был 1.00.
- **Цена**: ещё одна модель в памяти (~90 МБ весов) и 20 пар на прогон cross-encoder'ом при каждом запросе.

Включить: `RERANK_ENABLED=true` в `.env`. Параметры: `RERANK_MODEL`, `RERANK_FETCH_K` (по умолчанию 20), `RERANK_TOP_N` (по умолчанию 5).

Результаты прогона с reranker'ом: [notebooks/rag_metrics_rerank.json](notebooks/rag_metrics_rerank.json).

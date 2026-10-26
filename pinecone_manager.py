"""
Менеджер векторной памяти на базе Pinecone + OpenAI Embeddings.

Перед каждой записью проверяет, нет ли в индексе уже похожей информации:
если косинусное сходство >= SIMILARITY_THRESHOLD — это дубликат (слот обновляется
или запись пропускается), иначе — новая информация (вставляется новый вектор).

Если ключей нет или MOCK_MODE=true, работает в режиме эмуляции:
Pinecone заменяется InMemoryIndex, OpenAI-эмбеддинги — детерминированным mock_embedding.
"""

import hashlib
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv("local.env")

# Глобальная настройка порога сходства
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))  # >= порога это дубликат, < порога это новая информация
EMBEDDING_DIM = 1536

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "bot.log")


def setup_logging(level: int = logging.INFO) -> None:
    """Настраивает логирование в консоль и в файл logs/bot.log (один раз на процесс)."""
    root = logging.getLogger()
    if getattr(root, "_vpg05_configured", False):
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
    # Сторонние библиотеки слишком болтливы на INFO
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "pinecone", "TeleBot"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._vpg05_configured = True


setup_logging()
logger = logging.getLogger("pinecone_manager")


def env_flag(name: str, default: bool = False) -> bool:
    """Читает булев флаг из окружения: true/1/yes/on."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def is_real_value(value: Optional[str]) -> bool:
    """Значение задано и не является плейсхолдером из local.env.example."""
    return bool(value) and not str(value).strip().startswith("your_")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Достаёт поле и из dict, и из объекта ответа Pinecone."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Режим эмуляции: детерминированные эмбеддинги
# ---------------------------------------------------------------------------

# Маленький словарь «смысловых концептов» по основам слов. Он имитирует то, что
# настоящие эмбеддинги ловят синонимы: «хочу» и «полечу» разные строки,
# но оба про желание полететь, а «Марс» и «космос» — про одну тему.
MOCK_CONCEPTS: Dict[str, tuple] = {
    "космос": ("марс", "космо", "ракет", "планет", "звезд", "луна", "орбит"),
    "полет": ("лет", "леч", "полет", "улет"),
    "желание": ("хоч", "хот", "мечт"),
    "еда": ("пицц", "еда", "бургер", "суш", "вкусн"),
    "приветствие": ("привет", "здравств", "здорово"),
}


def _hash_feature(feature: str, dim: int) -> tuple:
    """Стабильный (не зависящий от запуска) хэш признака -> (индекс, знак)."""
    h = int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16)
    return h % dim, (1.0 if (h >> 64) & 1 else -1.0)


def mock_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Детерминированный эмбеддинг без OpenAI.

    Нижний регистр -> удаление пунктуации -> хэширование слов, символьных триграмм
    и смысловых концептов в вектор размерности dim -> L2-нормализация.
    Короткие служебные слова («я», «на», «и») получают маленький вес.
    """
    clean = text.lower().replace("ё", "е")
    clean = re.sub(r"[^\w\s]", " ", clean)
    words = clean.split()

    vec = np.zeros(dim, dtype=np.float64)

    def add(feature: str, weight: float) -> None:
        idx, sign = _hash_feature(feature, dim)
        vec[idx] += sign * weight

    for word in words:
        k = 0.25 if len(word) <= 2 else 1.0
        add("w:" + word, 1.5 * k)
        padded = f" {word} "
        for i in range(len(padded) - 2):
            add("t:" + padded[i:i + 3], k)
        for concept, stems in MOCK_CONCEPTS.items():
            if any(word.startswith(stem) for stem in stems):
                add("c:" + concept, 2.0)

    norm = np.linalg.norm(vec)
    if norm == 0:
        # Пустой текст: фиксированный единичный вектор, чтобы не делить на ноль
        vec[0] = 1.0
        norm = 1.0
    return (vec / norm).tolist()


# ---------------------------------------------------------------------------
# Режим эмуляции: индекс в памяти с интерфейсом Pinecone
# ---------------------------------------------------------------------------

def _match_filter(metadata: Dict[str, Any], flt: Optional[Dict[str, Any]]) -> bool:
    """Упрощённая поддержка фильтров Pinecone: равенство, $eq, $ne, $in, $nin, $and, $or."""
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not all(_match_filter(metadata, sub) for sub in cond):
                return False
            continue
        if key == "$or":
            if not any(_match_filter(metadata, sub) for sub in cond):
                return False
            continue
        value = metadata.get(key)
        if isinstance(cond, dict):
            for op, expected in cond.items():
                if op == "$eq" and value != expected:
                    return False
                if op == "$ne" and value == expected:
                    return False
                if op == "$in" and value not in expected:
                    return False
                if op == "$nin" and value in expected:
                    return False
        elif value != cond:
            return False
    return True


class InMemoryIndex:
    """Локальная замена индекса Pinecone. Ответы имеют ту же структуру, что у Pinecone."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        self._store: Dict[str, Dict[str, Any]] = {}

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = "") -> Dict[str, int]:
        for item in vectors:
            values = np.asarray(item["values"], dtype=np.float64)
            if values.shape[0] != self.dimension:
                raise ValueError(
                    f"Размерность вектора {values.shape[0]} не совпадает с индексом {self.dimension}"
                )
            self._store[item["id"]] = {
                "id": item["id"],
                "values": values,
                "metadata": dict(item.get("metadata") or {}),
            }
        return {"upserted_count": len(vectors)}

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True,
        include_values: bool = False,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "",
    ) -> Dict[str, Any]:
        q = np.asarray(vector, dtype=np.float64)
        q_norm = np.linalg.norm(q) or 1.0
        matches = []
        for item in self._store.values():
            if not _match_filter(item["metadata"], filter):
                continue
            v = item["values"]
            score = float(np.dot(q, v) / (q_norm * (np.linalg.norm(v) or 1.0)))
            match = {"id": item["id"], "score": score}
            if include_metadata:
                match["metadata"] = dict(item["metadata"])
            if include_values:
                match["values"] = v.tolist()
            matches.append(match)
        matches.sort(key=lambda m: m["score"], reverse=True)
        return {"matches": matches[:top_k], "namespace": namespace}

    def fetch(self, ids: List[str], namespace: str = "") -> Dict[str, Any]:
        vectors = {
            vid: {
                "id": vid,
                "values": self._store[vid]["values"].tolist(),
                "metadata": dict(self._store[vid]["metadata"]),
            }
            for vid in ids
            if vid in self._store
        }
        return {"vectors": vectors, "namespace": namespace}

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        delete_all: bool = False,
        namespace: str = "",
    ) -> Dict[str, Any]:
        if delete_all:
            self._store.clear()
        elif ids:
            for vid in ids:
                self._store.pop(vid, None)
        elif filter:
            for vid in [k for k, v in self._store.items() if _match_filter(v["metadata"], filter)]:
                self._store.pop(vid, None)
        return {}

    def describe_index_stats(self) -> Dict[str, Any]:
        count = len(self._store)
        return {
            "dimension": self.dimension,
            "index_fullness": 0.0,
            "total_vector_count": count,
            "namespaces": {"": {"vector_count": count}} if count else {},
        }

    def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "",
    ) -> Dict[str, Any]:
        if id not in self._store:
            raise KeyError(f"Вектор {id} не найден")
        if values is not None:
            self._store[id]["values"] = np.asarray(values, dtype=np.float64)
        if set_metadata:
            self._store[id]["metadata"].update(set_metadata)
        return {}


# ---------------------------------------------------------------------------
# Менеджер
# ---------------------------------------------------------------------------

class PineconeManager:
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_model: str = "text-embedding-3-small",
        similarity_threshold: Optional[float] = None,
        mock: Optional[bool] = None,
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "bot-memory")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL") or None
        self.openai_model = os.getenv("OPENAI_EMBED_MODEL") or openai_model
        self.similarity_threshold = (
            float(similarity_threshold) if similarity_threshold is not None else SIMILARITY_THRESHOLD
        )

        missing = [
            name
            for name, value in (("PINECONE_API_KEY", self.api_key), ("OPENAI_API_KEY", self.openai_api_key))
            if not is_real_value(value)
        ]
        if mock is None:
            mock = env_flag("MOCK_MODE") or bool(missing)
        self.mock = bool(mock)

        self.pc = None
        self.openai_client = None
        self._last_score: Optional[float] = None  # score ближайшего соседа последней проверки

        if self.mock:
            reason = "MOCK_MODE=true или mock=True" if not missing else f"нет ключей: {', '.join(missing)}"
            logger.warning("РЕЖИМ ЭМУЛЯЦИИ: OpenAI/Pinecone заменены локальными заглушками (%s)", reason)
            self.index = InMemoryIndex(EMBEDDING_DIM)
        else:
            self._init_openai()
            self._init_pinecone()

        logger.info(
            "PineconeManager готов | индекс: %s | порог сходства: %.3f | режим: %s",
            self.index_name,
            self.similarity_threshold,
            "эмуляция" if self.mock else "реальный",
        )

    # ----- инициализация клиентов -----

    def _init_openai(self) -> None:
        from openai import OpenAI

        try:
            if self.openai_base_url:
                self.openai_client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
            else:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI-клиент создан (base_url: %s, модель эмбеддингов: %s)",
                        self.openai_base_url or "по умолчанию", self.openai_model)
        except Exception as e:
            logger.error("Не удалось создать OpenAI-клиент: %s", e)
            raise

    def _init_pinecone(self) -> None:
        from pinecone import Pinecone, ServerlessSpec

        try:
            self.pc = Pinecone(api_key=self.api_key)
            existing = self.pc.list_indexes().names()
            if self.index_name not in existing:
                logger.info("Индекс «%s» не найден, создаю (dimension=%d, metric=cosine, aws/%s)",
                            self.index_name, EMBEDDING_DIM, self.environment)
                self.pc.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=self.environment),
                )
                self._wait_index_ready()
            else:
                logger.info("Индекс «%s» уже существует", self.index_name)
                desc = self.pc.describe_index(self.index_name)
                dim = _get(desc, "dimension")
                if dim and int(dim) != EMBEDDING_DIM:
                    logger.warning("Размерность индекса %s != %d — эмбеддинги не подойдут!", dim, EMBEDDING_DIM)
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            logger.error("Ошибка подключения к Pinecone: %s", e)
            raise

    def _wait_index_ready(self, timeout: int = 120) -> None:
        start = time.time()
        while time.time() - start < timeout:
            status = _get(self.pc.describe_index(self.index_name), "status")
            if _get(status, "ready", False):
                logger.info("Индекс «%s» готов", self.index_name)
                return
            time.sleep(2)
        logger.warning("Индекс «%s» не стал ready за %d с, продолжаю", self.index_name, timeout)

    # ----- эмбеддинги -----

    def create_embedding(self, text: str) -> List[float]:
        if self.mock:
            return mock_embedding(text)
        try:
            response = self.openai_client.embeddings.create(model=self.openai_model, input=text)
            vector = response.data[0].embedding
            if len(vector) != EMBEDDING_DIM:
                logger.warning("Модель вернула вектор размерности %d, ожидалось %d", len(vector), EMBEDDING_DIM)
            return vector
        except Exception as e:
            logger.error("Ошибка получения эмбеддинга от OpenAI: %s", e)
            raise

    # ----- проверка сходства -----

    def _check_similarity(self, vector: List[float], filter: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Ищет ближайший вектор. Возвращает его, если score >= порога, иначе None."""
        self._last_score = None
        try:
            response = self.index.query(vector=vector, top_k=1, include_metadata=True, filter=filter or None)
        except Exception as e:
            logger.error("Ошибка проверки сходства (query): %s", e)
            return None

        matches = _get(response, "matches", []) or []
        if not matches:
            self._last_score = 0.0
            logger.info("Проверка сходства: индекс пуст (или нет записей по фильтру) | score: 0.000")
            return None

        best = matches[0]
        score = float(_get(best, "score", 0.0))
        self._last_score = score
        best_id = _get(best, "id")
        is_dup = score >= self.similarity_threshold
        logger.info(
            "Проверка сходства: ближайший id=%s | score: %.3f | порог: %.3f -> %s",
            best_id, score, self.similarity_threshold, "ДУБЛИКАТ" if is_dup else "новая информация",
        )
        if is_dup:
            return {"id": best_id, "score": score, "metadata": dict(_get(best, "metadata", {}) or {})}
        return None

    # ----- запись -----

    def upsert_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        check_similarity: bool = True,
        on_duplicate: str = "update",
    ) -> Dict[str, Any]:
        metadata = {k: v for k, v in (metadata or {}).items() if v is not None and v != ""}
        similarity_score = None
        existing_id = None

        # Ищем дубликат в пределах того же пользователя, если он указан
        flt = {"user_id": metadata["user_id"]} if "user_id" in metadata else None
        duplicate = self._check_similarity(vector, filter=flt) if check_similarity else None

        if duplicate:
            similarity_score = duplicate["score"]
            existing_id = duplicate["id"]
            if on_duplicate == "skip":
                result = {"action": "skipped", "similarity_score": similarity_score,
                          "existing_id": existing_id, "id": existing_id}
                self._log_result(result)
                return result

            # on_duplicate == "update": перезаписываем существующий слот новыми данными
            old_meta = duplicate["metadata"]
            new_meta = dict(metadata)
            if "created_at" in old_meta:
                new_meta["created_at"] = old_meta["created_at"]
            new_meta["updated_at"] = now_iso()
            new_meta["mentions"] = int(old_meta.get("mentions", 1)) + 1
            target_id, action = existing_id, "updated"
        else:
            if check_similarity:
                similarity_score = self._last_score
            new_meta = dict(metadata)
            new_meta.setdefault("created_at", now_iso())
            new_meta["mentions"] = 1
            target_id, action = vector_id, "inserted"

        try:
            self.index.upsert(vectors=[{"id": target_id, "values": vector, "metadata": new_meta}])
        except Exception as e:
            logger.error("Ошибка записи вектора id=%s в индекс: %s", target_id, e)
            result = {"action": "error", "similarity_score": similarity_score,
                      "existing_id": existing_id, "id": target_id, "error": str(e)}
            self._log_result(result)
            return result

        result = {"action": action, "similarity_score": similarity_score,
                  "existing_id": existing_id, "id": target_id}
        self._log_result(result)
        return result

    def _log_result(self, result: Dict[str, Any]) -> None:
        score = result.get("similarity_score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        logger.info("action: %s | score: %s | id: %s", result["action"], score_str, result["id"])

    def upsert_vectors(self, items: List[Dict[str, Any]], check_similarity: bool = True,
                       on_duplicate: str = "update") -> List[Dict[str, Any]]:
        """items: [{"id", "values", "metadata"?}, ...]"""
        return [
            self.upsert_vector(item["id"], item["values"], item.get("metadata"),
                               check_similarity=check_similarity, on_duplicate=on_duplicate)
            for item in items
        ]

    def upsert_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        check_similarity: bool = True,
        on_duplicate: str = "update",
    ) -> Dict[str, Any]:
        try:
            vector = self.create_embedding(text)
        except Exception as e:
            result = {"action": "error", "similarity_score": None, "existing_id": None,
                      "id": doc_id, "error": str(e)}
            self._log_result(result)
            return result
        meta = dict(metadata or {})
        meta["text"] = text
        return self.upsert_vector(doc_id, vector, meta, check_similarity=check_similarity,
                                  on_duplicate=on_duplicate)

    def upsert_documents(self, docs: List[Dict[str, Any]], check_similarity: bool = True,
                         on_duplicate: str = "update") -> List[Dict[str, Any]]:
        """docs: [{"id", "text", "metadata"?}, ...]"""
        return [
            self.upsert_document(d["id"], d["text"], d.get("metadata"),
                                 check_similarity=check_similarity, on_duplicate=on_duplicate)
            for d in docs
        ]

    # ----- поиск -----

    def query_by_vector(self, vector: List[float], top_k: int = 5,
                        filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            response = self.index.query(vector=vector, top_k=top_k, include_metadata=True, filter=filter or None)
        except Exception as e:
            logger.error("Ошибка поиска по вектору: %s", e)
            return []
        return [
            {"id": _get(m, "id"), "score": float(_get(m, "score", 0.0)),
             "metadata": dict(_get(m, "metadata", {}) or {})}
            for m in (_get(response, "matches", []) or [])
        ]

    def query_by_text(self, text: str, top_k: int = 5,
                      filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            vector = self.create_embedding(text)
        except Exception:
            return []
        return self.query_by_vector(vector, top_k=top_k, filter=filter)

    # ----- служебные операции -----

    def fetch_vectors(self, ids: List[str]) -> Dict[str, Any]:
        try:
            response = self.index.fetch(ids=ids)
        except Exception as e:
            logger.error("Ошибка fetch для %s: %s", ids, e)
            return {}
        vectors = _get(response, "vectors", {}) or {}
        return {
            vid: {"id": vid, "values": list(_get(v, "values", []) or []),
                  "metadata": dict(_get(v, "metadata", {}) or {})}
            for vid, v in vectors.items()
        }

    def delete(self, ids: List[str]) -> bool:
        try:
            self.index.delete(ids=ids)
            logger.info("Удалены векторы: %s", ids)
            return True
        except Exception as e:
            logger.error("Ошибка удаления %s: %s", ids, e)
            return False

    def delete_by_filter(self, filter: Dict[str, Any]) -> bool:
        try:
            self.index.delete(filter=filter)
            logger.info("Удалены векторы по фильтру %s", filter)
            return True
        except Exception as e:
            # Некоторые конфигурации serverless не поддерживают delete по фильтру — удаляем через поиск id
            logger.warning("delete по фильтру не сработал (%s), удаляю через поиск id", e)
        try:
            zero = [0.0] * EMBEDDING_DIM
            zero[0] = 1.0
            ids = [m["id"] for m in self.query_by_vector(zero, top_k=1000, filter=filter)]
            if ids:
                self.index.delete(ids=ids)
            logger.info("Удалено %d векторов по фильтру %s", len(ids), filter)
            return True
        except Exception as e:
            logger.error("Ошибка удаления по фильтру %s: %s", filter, e)
            return False

    def delete_all(self) -> bool:
        try:
            self.index.delete(delete_all=True)
            logger.info("Индекс очищен (delete_all)")
            return True
        except Exception as e:
            # Pinecone serverless отвечает 404 «Namespace not found», если индекс уже пуст
            if "not found" in str(e).lower() or "404" in str(e):
                logger.info("Индекс уже пуст — очищать нечего")
                return True
            logger.error("Ошибка очистки индекса: %s", e)
            return False

    def describe_index_stats(self) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
        except Exception as e:
            logger.error("Ошибка получения статистики индекса: %s", e)
            return {}
        if hasattr(stats, "to_dict"):
            return stats.to_dict()
        return dict(stats)

    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        clean = {k: v for k, v in metadata.items() if v is not None and v != ""}
        try:
            self.index.update(id=vector_id, set_metadata=clean)
            logger.info("Обновлены метаданные id=%s: %s", vector_id, clean)
            return True
        except Exception as e:
            logger.error("Ошибка обновления метаданных id=%s: %s", vector_id, e)
            return False


# ---------------------------------------------------------------------------
# Ручной тест менеджера
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    manager = PineconeManager()

    print("\n=== 1. Проверка подключения: describe_index_stats() ===")
    print(manager.describe_index_stats())

    print("\n=== 2. Очистка индекса: delete_all() ===")
    manager.delete_all()
    if not manager.mock:
        time.sleep(2)

    phrases = [
        "Привет",
        "Зачем люди хотят лететь на Марс?",
        "Я бы хотел полететь на Марс",
        "Я бы не хотел полететь на Марс",
        "Хочу на Марс",
        "Я полечу на Марс",
        "Я люблю пиццу",
    ]

    print("\n=== 3. Последовательная запись документов (user_id=test) ===")
    rows = []
    for i, phrase in enumerate(phrases, start=1):
        res = manager.upsert_document(f"test_{i}", phrase, metadata={"user_id": "test", "source": "manual_test"})
        rows.append((phrase, res))
        if not manager.mock:
            time.sleep(1)  # Pinecone индексирует запись не мгновенно

    print("\n=== 4. Результаты ===")
    print(f"{'Фраза':<34} | {'action':<9} | {'score':<6} | id")
    print("-" * 66)
    for phrase, res in rows:
        score = res.get("similarity_score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        print(f"{phrase:<34} | {res['action']:<9} | {score_str:<6} | {res['id']}")

    print("\n=== 5. query_by_text('космос и Марс', top_k=3) ===")
    for m in manager.query_by_text("космос и Марс", top_k=3, filter={"user_id": "test"}):
        print(f"  {m['score']:.3f} | {m['id']} | {m['metadata'].get('text')}")

    print("\nСтатистика после теста:", manager.describe_index_stats())

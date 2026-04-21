from __future__ import annotations
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi


class KnowledgeBaseRetriever:
    def __init__(self, kb_path: str | Path) -> None:
        self._documents: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._load(Path(kb_path))

    def _load(self, path: Path) -> None:
        with open(path) as f:
            kb = json.load(f)

        docs: list[str] = []

        docs.append(f"AutoStream: {kb['company']['description']}")

        for plan in kb["plans"]:
            features = ", ".join(plan["features"])
            docs.append(
                f"Pricing — {plan['name']}: price is {plan['price']}, costs {plan['price']}. "
                f"This plan includes: {features}."
            )

        for policy in kb["policies"]:
            docs.append(f"{policy['title']}: {policy['content']}")

        for faq in kb.get("faq", []):
            docs.append(f"Q: {faq['question']} A: {faq['answer']}")

        self._documents = docs
        tokenized = [self._tokenize(d) for d in docs]
        self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"\w+", text.lower())
        # Light stemming: collapse plurals so "cost"/"costs", "plan"/"plans" match.
        return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens]

    def search(self, query: str, top_k: int = 5) -> str:
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = [self._documents[i] for i in ranked[:top_k] if scores[i] > 0]

        if not hits:
            # Small KB fallback: return everything
            return "\n\n".join(self._documents)

        return "\n\n".join(hits)


_retriever: KnowledgeBaseRetriever | None = None


def get_retriever() -> KnowledgeBaseRetriever:
    global _retriever
    if _retriever is None:
        kb_path = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"
        _retriever = KnowledgeBaseRetriever(kb_path)
    return _retriever

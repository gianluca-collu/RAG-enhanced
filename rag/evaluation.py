"""Small offline-by-dataset evaluator for retrieval quality."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Iterable

from llama_index.core.vector_stores.types import VectorStoreQueryMode

from rag.config import Config
from rag.rag_client import RagClient


def _normalize_section(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def retrieval_metrics(
    retrieved_sections: Iterable[str],
    relevant_sections: Iterable[str],
    k: int,
) -> dict[str, float]:
    """Return recall@k and reciprocal rank for exact expected section paths."""
    if k < 1:
        raise ValueError("k must be at least 1")
    relevant = {
        _normalize_section(value)
        for value in relevant_sections
        if isinstance(value, str) and value.strip()
    }
    if not relevant:
        raise ValueError("At least one relevant section is required")

    retrieved = [_normalize_section(value) for value in retrieved_sections]
    top_k = retrieved[:k]
    matched = set(top_k) & relevant
    reciprocal_rank = next(
        (1 / rank for rank, section in enumerate(top_k, start=1) if section in relevant),
        0.0,
    )
    return {
        f"recall@{k}": len(matched) / len(relevant),
        f"mrr@{k}": reciprocal_rank,
    }


def evaluate_retrieval(dataset_path: Path, config: Config, top_k: int = 5) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON list")

    client = RagClient(config)
    mode = VectorStoreQueryMode.HYBRID if config.hybrid_search else VectorStoreQueryMode.DEFAULT
    retriever = client.index.as_retriever(
        similarity_top_k=top_k,
        vector_store_query_mode=mode,
    )

    results = []
    for row_number, case in enumerate(dataset, start=1):
        question = case.get("question")
        expected = case.get("relevant_sections")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Case {row_number}: question must be a non-empty string")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"Case {row_number}: relevant_sections must be a non-empty list")

        nodes = retriever.retrieve(question)
        retrieved_sections = [
            str(node.node.metadata.get("section_path", ""))
            for node in nodes
            if node.node.metadata.get("section_path")
        ]
        results.append({
            "question": question,
            "retrieved_sections": retrieved_sections,
            **retrieval_metrics(retrieved_sections, expected, top_k),
        })

    return {
        "case_count": len(results),
        "top_k": top_k,
        "mode": mode.value,
        f"mean_recall@{top_k}": mean(result[f"recall@{top_k}"] for result in results),
        f"mean_mrr@{top_k}": mean(result[f"mrr@{top_k}"] for result in results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Azure AI Search retrieval against expected sections.")
    parser.add_argument("dataset", type=Path, help="JSON file with question and relevant_sections cases")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("output/retrieval_eval.json"))
    args = parser.parse_args()

    report = evaluate_retrieval(args.dataset, Config.from_env(str(args.env)), top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()

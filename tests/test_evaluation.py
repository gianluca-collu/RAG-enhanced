import pytest

from rag.evaluation import retrieval_metrics


def test_retrieval_metrics_normalizes_sections_and_computes_recall_and_mrr():
    metrics = retrieval_metrics(
        retrieved_sections=[" Bando > Art. 2 ", "bando > art. 1"],
        relevant_sections=["bando > art. 1", "bando > art. 3"],
        k=2,
    )

    assert metrics == {"recall@2": 0.5, "mrr@2": 0.5}


def test_retrieval_metrics_requires_expected_sections():
    with pytest.raises(ValueError, match="relevant section"):
        retrieval_metrics(["Bando > Art. 1"], [], k=5)

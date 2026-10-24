"""The retrieval benchmark must never count unanswerable questions as misses."""
import pytest
from evaluate import EvalItem, retrieval_hit


class FakeRetriever:
    def retrieve(self, question, k):
        return []


class FakePipeline:
    _retriever = FakeRetriever()


def test_out_of_scope_question_excluded_from_retrieval_metric():
    item = EvalItem("Outside the PDF?", "Not present", "example.pdf", 0)
    assert not item.answerable
    with pytest.raises(ValueError, match="answerable"):
        retrieval_hit(FakePipeline(), item)


def test_answerable_question_is_counted_normally():
    item = EvalItem("Inside the PDF?", "Answer", "example.pdf", 3)
    assert item.answerable
    assert retrieval_hit(FakePipeline(), item) is False

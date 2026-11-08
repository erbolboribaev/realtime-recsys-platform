import numpy as np
import pytest

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.ml.als import ALSModel
from recsys.ml.dataset import build_interactions, deduplicate, future_items, time_split
from recsys.ml.index import NumpyItemIndex
from recsys.ml.metrics import evaluate
from recsys.ml.retrieval import Retriever
from recsys.ml.tracking import sanitize_metric_name, uses_file_store
from recsys.ml.train import index_agreement


def fitted():
    config = SimulatorConfig(n_users=120, n_items=250, n_categories=6, seed=5)
    events = [e.model_dump() for e in EventSimulator(config).take(9_000)]
    split = time_split(deduplicate(events), test_fraction=0.2)
    inter = build_interactions(split.train)
    return ALSModel(factors=8, iterations=4).fit(inter), inter, future_items(split.test, inter)


def test_numpy_index_orders_by_dot_product_and_honours_exclusions():
    index = NumpyItemIndex()
    index.build(np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]), ["a", "b", "c"])
    assert index.search(np.array([1.0, 0.0]), k=2) == [2, 0]
    assert index.search(np.array([1.0, 0.0]), k=2, exclude=[2]) == [0, 1]


def test_searching_an_unbuilt_index_fails_clearly():
    with pytest.raises(RuntimeError, match="build"):
        NumpyItemIndex().search(np.array([1.0]), k=1)


def test_retrieval_through_the_index_equals_the_model_ranking():
    model, inter, _ = fitted()
    assert index_agreement(model, inter, NumpyItemIndex(), k=20) == 1.0


def test_retrieval_gives_the_same_offline_metrics_as_the_model():
    model, inter, truth = fitted()
    retriever = Retriever(model, inter, NumpyItemIndex())
    retriever.publish()
    assert evaluate(retriever.recommend, truth, 20) == evaluate(model.recommend, truth, 20)


def test_retriever_requires_a_fitted_model():
    _, inter, _ = fitted()
    with pytest.raises(ValueError, match="fitted"):
        Retriever(ALSModel(), inter, NumpyItemIndex())


def test_metric_names_are_made_safe_for_mlflow():
    assert sanitize_metric_name("als_recall@50") == "als_recall_at_50"


def test_only_folders_and_file_uris_count_as_the_file_store():
    assert uses_file_store("./mlruns")
    assert uses_file_store("file:///tmp/mlruns")
    assert not uses_file_store("sqlite:///mlflow.db")
    assert not uses_file_store("http://localhost:5000")


def test_qdrant_index_matches_the_exact_reference():
    qdrant_client = pytest.importorskip("qdrant_client")
    from recsys.ml.index import QdrantItemIndex

    model, inter, _ = fitted()
    index = QdrantItemIndex(qdrant_client.QdrantClient(":memory:"), "items-test")
    assert index_agreement(model, inter, index, k=20) >= 0.95


def test_mlflow_run_is_recorded(tmp_path):
    pytest.importorskip("mlflow")
    import mlflow

    from recsys.ml.tracking import log_run

    uri = (tmp_path / "mlruns").as_uri()
    run_id = log_run("test-exp", "run-1", {"factors": 8}, {"recall@50": 0.25}, uri)
    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["recall_at_50"] == 0.25
    assert run.data.params["factors"] == "8"

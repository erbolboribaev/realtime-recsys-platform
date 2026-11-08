import pytest

from recsys.sink.consumer import ensure_bucket


class FakeFs:
    """Stand-in for an fsspec filesystem whose backend may still be starting up."""

    def __init__(self, has_bucket: bool = False, fail_times: int = 0) -> None:
        self.has_bucket = has_bucket
        self.fail_times = fail_times
        self.created: list[str] = []

    def exists(self, bucket: str) -> bool:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("connection refused")
        return self.has_bucket

    def mkdir(self, bucket: str) -> None:
        self.created.append(bucket)
        self.has_bucket = True


def test_creates_the_bucket_when_it_is_missing():
    fs = FakeFs(has_bucket=False)
    ensure_bucket(fs, "recsys-lake", sleep=lambda _s: None)
    assert fs.created == ["recsys-lake"]


def test_leaves_an_existing_bucket_alone():
    fs = FakeFs(has_bucket=True)
    ensure_bucket(fs, "recsys-lake", sleep=lambda _s: None)
    assert fs.created == []


def test_retries_while_the_store_is_starting():
    fs = FakeFs(fail_times=3)
    sleeps: list[float] = []
    ensure_bucket(fs, "recsys-lake", delay_s=2.0, sleep=sleeps.append)
    assert sleeps == [2.0, 2.0, 2.0]
    assert fs.created == ["recsys-lake"]


def test_gives_up_with_a_clear_error_after_all_attempts():
    fs = FakeFs(fail_times=100)
    with pytest.raises(RuntimeError, match="not reachable after 5 attempts"):
        ensure_bucket(fs, "recsys-lake", attempts=5, sleep=lambda _s: None)

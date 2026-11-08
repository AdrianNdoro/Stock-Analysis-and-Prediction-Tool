import threading

from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.tests.fakes import FakeClock


def test_roundtrip_upsert_delete(tmp_path):
    clock = FakeClock(1000.0)
    c = SQLiteCache(tmp_path / "c.sqlite3", clock=clock)
    assert c.get("k") is None
    c.set("k", "v1")
    assert c.get("k").value == "v1"
    clock.advance(10)
    c.set("k", "v2")
    e = c.get("k")
    assert e.value == "v2" and e.stored_at == 1010.0
    c.delete("k")
    assert c.get("k") is None


def test_freshness_and_purge(tmp_path):
    clock = FakeClock(1000.0)
    c = SQLiteCache(tmp_path / "c.sqlite3", clock=clock)
    c.set("old", "x")
    clock.advance(100)
    c.set("new", "y")
    assert c.is_fresh(c.get("new"), 50) and not c.is_fresh(c.get("old"), 50)
    assert c.purge_older_than(50) == 1
    assert c.get("old") is None and c.get("new") is not None


def test_persists_across_instances_and_uses_wal(tmp_path):
    path = tmp_path / "c.sqlite3"
    SQLiteCache(path).set("k", "v")
    assert SQLiteCache(path).get("k").value == "v"
    import sqlite3

    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_parallel_readers_and_writers_do_not_corrupt(tmp_path):
    c = SQLiteCache(tmp_path / "c.sqlite3")
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            for n in range(25):
                c.set(f"key{i % 4}", f"{i}-{n}" * 200)  # contended keys
                c.set(f"own{i}", str(n))
                assert c.get(f"key{i % 4}").value  # never a partial/None value after a write
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert all(c.get(f"own{i}").value == "24" for i in range(12))

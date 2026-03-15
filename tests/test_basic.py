"""Basic round-trip tests for LMDBMixin."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from pydantic import BaseModel
from typing import Any

from lmdb_mixin import LMDBMixin


# -------------------------------------------------------------------
# Fixture: temp directory for LMDB files
# -------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "test.lmdb")
    yield path
    LMDBMixin.close_env(path)


# -------------------------------------------------------------------
# Model definitions
# -------------------------------------------------------------------

class EmbeddingResult(BaseModel):
    """Inner model – does NOT inherit LMDBMixin."""
    label: str
    score: float
    vector: Any


class ExperimentRecord(LMDBMixin):
    """Top-level model – inherits LMDBMixin."""
    name: str
    tags: list[str]
    result: EmbeddingResult
    weight_matrix: Any
    config: dict


class SimpleModel(LMDBMixin):
    """Minimal model with only JSON-safe fields."""
    name: str
    value: int
    active: bool


class NdarrayModel(LMDBMixin):
    """Model with ndarray fields only."""
    vec: Any
    mat: Any


class PickleModel(LMDBMixin):
    """Model with a pickle-fallback field."""
    data: Any


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestBasicRoundTrip:
    """Store → load → compare for all supported types."""

    def test_simple_json_fields(self, db_path):
        record = SimpleModel(name="hello", value=42, active=True)
        key = record.store(db_path)
        restored = SimpleModel.load(key, db_path)
        assert restored.name == "hello"
        assert restored.value == 42
        assert restored.active is True

    def test_ndarray_round_trip(self, db_path):
        vec = np.random.randn(768).astype("float32")
        mat = np.eye(4, dtype="float64")
        record = NdarrayModel(vec=vec, mat=mat)
        key = record.store(db_path)
        restored = NdarrayModel.load(key, db_path)
        np.testing.assert_array_equal(restored.vec, vec)
        np.testing.assert_array_equal(restored.mat, mat)

    def test_full_experiment_record(self, db_path):
        vec = np.random.randn(768).astype("float32")
        wm = np.eye(4)
        record = ExperimentRecord(
            name="run_42",
            tags=["nlp", "encoder"],
            result=EmbeddingResult(label="bert", score=0.987, vector=vec),
            weight_matrix=wm,
            config={"lr": 1e-4, "epochs": 10},
        )
        key = record.store(db_path)
        restored = ExperimentRecord.load(key, db_path)

        assert restored.name == "run_42"
        assert restored.tags == ["nlp", "encoder"]
        assert restored.result.label == "bert"
        assert restored.result.score == pytest.approx(0.987)
        np.testing.assert_array_equal(restored.result.vector, vec)
        np.testing.assert_array_equal(restored.weight_matrix, wm)
        assert restored.config == {"lr": 1e-4, "epochs": 10}

    def test_pickle_fallback(self, db_path):
        """Arbitrary non-JSON-serializable object should be pickled."""
        data = {"nested": {1, 2, 3}}  # set is not JSON-native
        record = PickleModel(data=data)
        key = record.store(db_path)
        restored = PickleModel.load(key, db_path)
        assert restored.data == data

    def test_user_supplied_key(self, db_path):
        record = SimpleModel(name="key_test", value=1, active=False)
        key = record.store(db_path, key="my_key")
        assert key == "my_key"
        restored = SimpleModel.load("my_key", db_path)
        assert restored.name == "key_test"

    def test_auto_uuid_key(self, db_path):
        record = SimpleModel(name="uuid_test", value=2, active=True)
        key = record.store(db_path)
        # key should look like a UUID4
        import uuid as _uuid
        _uuid.UUID(key, version=4)

    def test_key_not_found_raises(self, db_path):
        # Ensure env is initialised
        SimpleModel(name="x", value=0, active=True).store(db_path)
        with pytest.raises(KeyError, match="not_exist"):
            SimpleModel.load("not_exist", db_path)


class TestNestedAndComplex:
    """Tests for nesting depth, list of ndarrays, etc."""

    def test_list_of_ndarrays(self, db_path):
        class ArrayListModel(LMDBMixin):
            arrays: Any

        arrs = [np.arange(i * 3, (i + 1) * 3, dtype="float32") for i in range(5)]
        record = ArrayListModel(arrays=arrs)
        key = record.store(db_path)
        restored = ArrayListModel.load(key, db_path)
        for orig, rest in zip(arrs, restored.arrays):
            np.testing.assert_array_equal(rest, orig)

    def test_deeply_nested_dict(self, db_path):
        class DeepModel(LMDBMixin):
            payload: dict

        payload = {
            "level1": {
                "level2": {
                    "vec": np.zeros(4, dtype="float32"),
                    "val": 42,
                }
            }
        }
        record = DeepModel(payload=payload)
        key = record.store(db_path)
        restored = DeepModel.load(key, db_path)
        np.testing.assert_array_equal(
            restored.payload["level1"]["level2"]["vec"],
            payload["level1"]["level2"]["vec"],
        )
        assert restored.payload["level1"]["level2"]["val"] == 42

    def test_none_field(self, db_path):
        class OptionalModel(LMDBMixin):
            data: Any = None

        record = OptionalModel()
        key = record.store(db_path)
        restored = OptionalModel.load(key, db_path)
        assert restored.data is None

    def test_empty_ndarray(self, db_path):
        record = NdarrayModel(vec=np.array([], dtype="float32"), mat=np.zeros((0, 3)))
        key = record.store(db_path)
        restored = NdarrayModel.load(key, db_path)
        assert restored.vec.shape == (0,)
        assert restored.mat.shape == (0, 3)

    def test_high_dim_ndarray(self, db_path):
        arr = np.random.randn(2, 3, 4, 5).astype("float32")
        record = NdarrayModel(vec=arr, mat=np.array(1.0))
        key = record.store(db_path)
        restored = NdarrayModel.load(key, db_path)
        np.testing.assert_array_equal(restored.vec, arr)


class TestEnvCache:
    """Tests for the LMDB environment cache."""

    def test_env_cache_reuse(self, db_path):
        record = SimpleModel(name="a", value=1, active=True)
        record.store(db_path)
        assert db_path in LMDBMixin._env_cache
        # Second store should reuse the same env object.
        env1 = LMDBMixin._env_cache[db_path]
        record.store(db_path)
        env2 = LMDBMixin._env_cache[db_path]
        assert env1 is env2

    def test_close_env(self, db_path):
        SimpleModel(name="a", value=1, active=True).store(db_path)
        assert db_path in LMDBMixin._env_cache
        LMDBMixin.close_env(db_path)
        assert db_path not in LMDBMixin._env_cache

    def test_close_all_envs(self, tmp_path):
        p1 = str(tmp_path / "a.lmdb")
        p2 = str(tmp_path / "b.lmdb")
        SimpleModel(name="a", value=1, active=True).store(p1)
        SimpleModel(name="b", value=2, active=False).store(p2)
        assert p1 in LMDBMixin._env_cache
        assert p2 in LMDBMixin._env_cache
        LMDBMixin.close_all_envs()
        assert len(LMDBMixin._env_cache) == 0


class TestMultipleRecords:
    """Multiple store/load calls in the same DB."""

    def test_multiple_keys(self, db_path):
        records = [
            SimpleModel(name=f"rec_{i}", value=i, active=i % 2 == 0)
            for i in range(10)
        ]
        keys = [r.store(db_path) for r in records]
        for key, orig in zip(keys, records):
            restored = SimpleModel.load(key, db_path)
            assert restored.name == orig.name
            assert restored.value == orig.value

    def test_overwrite_same_key(self, db_path):
        r1 = SimpleModel(name="first", value=1, active=True)
        r1.store(db_path, key="shared")
        r2 = SimpleModel(name="second", value=2, active=False)
        r2.store(db_path, key="shared")
        restored = SimpleModel.load("shared", db_path)
        assert restored.name == "second"
        assert restored.value == 2


class TestEdgeCases:
    """Edge cases and known limitations."""

    def test_tuple_degrades_to_list(self, db_path):
        """JSON has no tuple type, so tuples come back as lists."""
        class TupleModel(LMDBMixin):
            data: Any

        record = TupleModel(data=(1, 2, 3))
        key = record.store(db_path)
        restored = TupleModel.load(key, db_path)
        assert restored.data == [1, 2, 3]

    def test_ndarray_dtype_preservation(self, db_path):
        """Various dtypes should survive the round trip."""
        for dt in ["float32", "float64", "int32", "int64", "complex64", "bool"]:
            arr = np.array([1, 0, 1], dtype=dt)
            record = NdarrayModel(vec=arr, mat=arr)
            key = record.store(db_path)
            restored = NdarrayModel.load(key, db_path)
            assert restored.vec.dtype == np.dtype(dt)
            np.testing.assert_array_equal(restored.vec, arr)

    def test_restored_ndarray_is_writable(self, db_path):
        """Restored array must be a writable copy (not an LMDB mmap view)."""
        arr = np.arange(5, dtype="float32")
        record = NdarrayModel(vec=arr, mat=arr)
        key = record.store(db_path)
        restored = NdarrayModel.load(key, db_path)
        restored.vec[0] = 999.0  # should not raise
        assert restored.vec[0] == 999.0

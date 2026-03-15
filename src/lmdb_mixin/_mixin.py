"""LMDBMixin – transparent LMDB persistence for Pydantic models.

Only the **top-level** Pydantic model needs to inherit ``LMDBMixin``.
Nested sub-models remain plain ``BaseModel`` – the serialisation logic
traverses them transparently via ``model_dump()``.
"""

from __future__ import annotations

import json
import pickle
import struct
import uuid
from typing import Any, ClassVar

import lmdb
import numpy as np
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAP_SIZE: int = 10 * (1024 ** 3)  # 10 GiB
_SENTINEL_KEY = "__lmdb_ref__"
_SENTINEL_TYPE = "__type__"
_CLASS_KEY = "__pydantic_class__"

# ---------------------------------------------------------------------------
# LMDBMixin
# ---------------------------------------------------------------------------


class LMDBMixin(BaseModel):
    """Mixin that gives any Pydantic model atomic LMDB persistence.

    Usage::

        class MyRecord(LMDBMixin):
            name: str
            vector: Any  # e.g. numpy ndarray

        key = MyRecord(name="x", vector=np.zeros(3)).store("/data/db.lmdb")
        restored = MyRecord.load(key, "/data/db.lmdb")
    """

    # Shared LMDB environment cache ------------------------------------------
    _env_cache: ClassVar[dict[str, lmdb.Environment]] = {}

    # Allow numpy arrays and other arbitrary types in fields
    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    @classmethod
    def _get_env(cls, path: str, map_size: int = _DEFAULT_MAP_SIZE) -> lmdb.Environment:
        """Return a cached ``lmdb.Environment`` for *path*, opening if needed."""
        env = cls._env_cache.get(path)
        if env is not None:
            return env
        env = lmdb.open(path, map_size=map_size, max_dbs=0)
        cls._env_cache[path] = env
        return env

    @classmethod
    def close_env(cls, path: str) -> None:
        """Close and evict the cached environment for *path*."""
        env = cls._env_cache.pop(path, None)
        if env is not None:
            env.close()

    @classmethod
    def close_all_envs(cls) -> None:
        """Close every cached environment (call before process exit)."""
        for env in cls._env_cache.values():
            env.close()
        cls._env_cache.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        path: str,
        key: str | None = None,
        *,
        map_size: int = _DEFAULT_MAP_SIZE,
    ) -> str:
        """Serialise this model into LMDB at *path* and return the key.

        Parameters
        ----------
        path : str
            Filesystem path of the LMDB database directory.
        key : str, optional
            User-defined key.  A UUID4 is generated when omitted.
        map_size : int
            ``map_size`` forwarded to ``lmdb.open`` on first access.

        Returns
        -------
        str
            The key under which the JSON envelope was stored.
        """
        if key is None:
            key = str(uuid.uuid4())

        env = self._get_env(path, map_size=map_size)

        # model_dump() recursively expands nested Pydantic models to dicts.
        raw = self.model_dump()

        with env.begin(write=True) as txn:
            payload = self._serialize_node(raw, txn)
            # Inject the fully-qualified class name so load() is self-describing.
            payload[_CLASS_KEY] = f"{type(self).__module__}.{type(self).__qualname__}"
            txn.put(key.encode("utf-8"), json.dumps(payload).encode("utf-8"))

        return key

    @classmethod
    def load(
        cls,
        key: str,
        path: str,
        *,
        map_size: int = _DEFAULT_MAP_SIZE,
    ) -> "LMDBMixin":
        """Reconstruct a model instance from LMDB.

        Parameters
        ----------
        key : str
            The key returned by a prior ``store()`` call.
        path : str
            Filesystem path of the LMDB database directory.
        map_size : int
            ``map_size`` forwarded to ``lmdb.open`` on first access.

        Returns
        -------
        LMDBMixin
            A fully validated Pydantic instance of this class.
        """
        env = cls._get_env(path, map_size=map_size)

        with env.begin(write=False) as txn:
            raw = txn.get(key.encode("utf-8"))
            if raw is None:
                raise KeyError(f"Key {key!r} not found in LMDB at {path!r}")
            data = json.loads(raw.decode("utf-8"))
            data.pop(_CLASS_KEY, None)
            data = cls._deserialize_node(data, txn)

        return cls(**data)

    # ------------------------------------------------------------------
    # Serialisation (recursive, runs inside a write txn)
    # ------------------------------------------------------------------

    @classmethod
    def _serialize_node(cls, node: Any, txn: lmdb.Transaction) -> Any:
        """Recursively walk *node* and replace non-JSON-safe values with
        sentinel reference nodes stored as separate LMDB entries.
        """
        if isinstance(node, dict):
            return {k: cls._serialize_node(v, txn) for k, v in node.items()}

        if isinstance(node, (list, tuple)):
            return [cls._serialize_node(item, txn) for item in node]

        # numpy ndarray → dedicated binary blob
        if isinstance(node, np.ndarray):
            return cls._put_ndarray(node, txn)

        # JSON-native primitives pass through.
        if node is None or isinstance(node, (bool, int, float, str)):
            return node

        # Anything else: try JSON-encoding; fall back to pickle.
        try:
            json.dumps(node)
            return node
        except (TypeError, ValueError, OverflowError):
            return cls._put_pickle(node, txn)

    @classmethod
    def _deserialize_node(cls, node: Any, txn: lmdb.Transaction) -> Any:
        """Recursively walk *node* and replace sentinel reference nodes with
        their materialised values.
        """
        if isinstance(node, dict):
            if _SENTINEL_KEY in node:
                return cls._fetch_ref(node, txn)
            return {k: cls._deserialize_node(v, txn) for k, v in node.items()}

        if isinstance(node, list):
            return [cls._deserialize_node(item, txn) for item in node]

        return node

    # ------------------------------------------------------------------
    # Blob helpers
    # ------------------------------------------------------------------

    @classmethod
    def _put_ndarray(cls, arr: np.ndarray, txn: lmdb.Transaction) -> dict:
        """Store a numpy array as a binary blob and return a sentinel dict."""
        ref_id = str(uuid.uuid4())

        # Ensure C-contiguous memory layout.
        arr = np.ascontiguousarray(arr)

        meta = json.dumps({"dtype": str(arr.dtype), "shape": list(arr.shape)}).encode("utf-8")
        meta_len = struct.pack(">I", len(meta))
        blob = meta_len + meta + arr.tobytes()
        txn.put(ref_id.encode("utf-8"), blob)

        return {_SENTINEL_KEY: ref_id, _SENTINEL_TYPE: "ndarray"}

    @classmethod
    def _put_pickle(cls, obj: Any, txn: lmdb.Transaction) -> dict:
        """Pickle *obj* into LMDB and return a sentinel dict."""
        ref_id = str(uuid.uuid4())
        blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        txn.put(ref_id.encode("utf-8"), blob)
        return {_SENTINEL_KEY: ref_id, _SENTINEL_TYPE: "pickle"}

    @classmethod
    def _fetch_ref(cls, sentinel: dict, txn: lmdb.Transaction) -> Any:
        """Materialise a sentinel reference node into its original value."""
        ref_id: str = sentinel[_SENTINEL_KEY]
        ref_type: str = sentinel[_SENTINEL_TYPE]
        raw = txn.get(ref_id.encode("utf-8"))
        if raw is None:
            raise KeyError(f"Blob {ref_id!r} (type={ref_type}) missing from LMDB")

        if ref_type == "ndarray":
            return cls._load_ndarray(raw)
        if ref_type == "pickle":
            return pickle.loads(raw)

        raise ValueError(f"Unknown sentinel type {ref_type!r}")

    @staticmethod
    def _load_ndarray(raw: bytes) -> np.ndarray:
        """Decode the custom ndarray binary layout."""
        meta_len = struct.unpack(">I", raw[:4])[0]
        meta = json.loads(raw[4 : 4 + meta_len])
        data = raw[4 + meta_len :]
        arr = np.frombuffer(data, dtype=np.dtype(meta["dtype"]))
        arr = arr.reshape(meta["shape"])
        return arr.copy()  # own memory – detach from LMDB mmap

# LMDBMixin

Transparent LMDB persistence for Pydantic models.

`LMDBMixin` is a Pydantic `BaseModel` mixin that provides self-contained, atomic
persistence backed by LMDB.  Any top-level Pydantic model that inherits
`LMDBMixin` can be stored and restored with a single method call — no external
manager required.

## Features

- **Self-describing** — only the class definition and the LMDB path are needed to
  fully restore an object via `MyModel.load(key, path)`.
- **Heterogeneous fields** — numpy ndarrays, arbitrary pickle-able objects, and
  JSON-native types coexist in the same model transparently.
- **Atomic writes** — all blob entries and the JSON envelope are written in one
  LMDB transaction.
- **Low overhead** — LMDB environments are cached per-path for the process
  lifetime.
- **Minimal intrusion** — nested Pydantic sub-models do **not** need to inherit
  the mixin.

## Quick start

```python
from lmdb_mixin import LMDBMixin
import numpy as np
from typing import Any
from pydantic import BaseModel

class EmbeddingResult(BaseModel):
    label: str
    score: float
    vector: Any

class ExperimentRecord(LMDBMixin):
    name: str
    tags: list[str]
    result: EmbeddingResult
    weight_matrix: Any
    config: dict

# Store
record = ExperimentRecord(
    name="run_42",
    tags=["nlp", "encoder"],
    result=EmbeddingResult(
        label="bert", score=0.987,
        vector=np.random.randn(768).astype("f4"),
    ),
    weight_matrix=np.eye(4),
    config={"lr": 1e-4, "epochs": 10},
)
key = record.store("/data/runs.lmdb")

# Restore (only needs the class definition + path)
restored = ExperimentRecord.load(key, "/data/runs.lmdb")

# Cleanup
ExperimentRecord.close_all_envs()
```

## Installation

```bash
pip install lmdb-mixin
```

## Known limitations

| Item | Note |
|---|---|
| `tuple` degradation | JSON has no tuple type; tuples round-trip as lists |
| Cascade delete | `store()` does not provide `delete()`; collect `__lmdb_ref__` keys manually |
| LMDB `map_size` | Defaults to 10 GiB virtual address space (does not pre-allocate disk) |
| Concurrent writes | LMDB allows only one write transaction at a time |
| Pickle safety | Never load untrusted LMDB files — pickle deserialisation can execute code |

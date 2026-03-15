# LMDBMixin 设计文档

## 概述

`LMDBMixin` 是一个 Pydantic `BaseModel` 的 Mixin 类，为任意顶层 Pydantic 数据结构提供基于 LMDB 的自洽持久化能力。继承该 Mixin 的类无需外部管理器，仅凭类型定义与 LMDB 文件路径即可完成存储与还原。

---

## 设计目标

| 目标 | 说明 |
|---|---|
| 自解析 | 只要知道类定义与 lmdb 路径，即可调用 `MyModel.load(key, path)` 完整还原对象 |
| 异构字段透明存储 | numpy ndarray、任意 pickle 对象与 JSON 原生类型共存于同一模型，无需手动区分 |
| 原子性 | 一次 `store()` 内所有写操作（子 blob + JSON envelope）在同一 LMDB 写事务中提交 |
| 低开销 | LMDB 环境按路径缓存，进程生命周期内只打开一次 |
| 最小侵入 | 内层嵌套 Pydantic 模型无需继承 Mixin，序列化逻辑透明穿透 |

---

## 架构

```
┌──────────────────────────────────────────────┐
│            用户定义的 Pydantic 模型            │
│         class Foo(LMDBMixin): ...             │
│                                               │
│  foo.store(path)   Foo.load(key, path)        │
└─────────────────┬──────────────────┬──────────┘
                  │                  │
        ┌─────────▼──────────────────▼─────────┐
        │              LMDBMixin                │
        │                                       │
        │  _serialize_node()   (递归，写事务内)  │
        │  _deserialize_node() (递归，读事务内)  │
        │  _env_cache: ClassVar[dict[str, Env]] │
        └─────────┬──────────────────┬──────────┘
                  │                  │
        ┌─────────▼──────────────────▼─────────┐
        │              LMDB 文件                │
        │                                       │
        │  key (用户指定或 UUID4)               │
        │    └─ JSON envelope  (顶层结构)        │
        │  <uuid4-1>  ndarray blob              │
        │  <uuid4-2>  pickle blob               │
        │  ...                                  │
        └───────────────────────────────────────┘
```

---

## 数据格式

### JSON Envelope（顶层条目）

`model_dump()` 递归展开后，所有可直接序列化的字段保持原样内联；不可序列化的字段被替换为 **Sentinel 引用节点**，最终整体编码为 UTF-8 JSON 字符串存入 LMDB。

```json
{
  "__pydantic_class__": "mymodule.ExperimentRecord",
  "name": "run_42",
  "tags": ["nlp", "encoder"],
  "config": {"lr": 0.0001, "epochs": 10},
  "result": {
    "label": "sentence_bert",
    "score": 0.9871,
    "vector": {
      "__lmdb_ref__": "a3f1c2d4-...",
      "__type__": "ndarray"
    }
  },
  "weight_matrix": {
    "__lmdb_ref__": "b7e9a1f0-...",
    "__type__": "ndarray"
  }
}
```

### Sentinel 引用节点

```json
{ "__lmdb_ref__": "<uuid4>", "__type__": "ndarray" | "pickle" }
```

Sentinel 节点本身是合法 JSON，可出现在任意嵌套层级（dict value、list 元素等）。

### ndarray Blob 布局

```
字节偏移   内容
─────────────────────────────────────────────────
[0, 4)     meta_len：4 字节大端无符号整数
[4, 4+L)   meta JSON（L = meta_len）：
             {"dtype": "<str>", "shape": [d0, d1, ...]}
[4+L, end) arr.tobytes()：原始数组字节（C-contiguous）
```

不依赖 `.npy` / HDF5 等外部格式；解析仅需 `struct.unpack` + `np.frombuffer`，适合高吞吐 HPC 场景。

### Pickle Blob

直接存储 `pickle.dumps(obj, protocol=HIGHEST_PROTOCOL)` 的字节流，覆盖 ndarray 以外的任意不可序列化类型（如 `torch.Tensor`、自定义类实例等）。

---

## 关键流程

### store()

```
model.model_dump()
    │
    ▼
_serialize_node(dict, txn)          ← 递归遍历
    ├── JSON 原生类型       → 原样内联
    ├── numpy.ndarray      → _put_ndarray()  → txn.put(uuid, blob)  → 返回 Sentinel
    ├── dict / list        → 递归
    └── 其他类型           → json.dumps() 探测
                               ├── 成功  → 原样内联
                               └── 失败  → _put_pickle() → txn.put(uuid, pkl) → 返回 Sentinel
    │
    ▼
注入 __pydantic_class__ 字段
    │
    ▼
txn.put(key, json.dumps(payload))   ← 所有写操作在同一事务内
    │
    ▼
返回 key (str)
```

### load()

```
txn.get(key)
    │
    ▼
json.loads(raw)
    │
    ▼
_deserialize_node(dict, txn)        ← 递归遍历（同一读事务）
    ├── 普通 dict / list  → 递归
    └── Sentinel 节点     → _fetch_ref()
                               ├── ndarray → struct.unpack + np.frombuffer + .copy()
                               └── pickle  → pickle.loads()
    │
    ▼
data.pop("__pydantic_class__")
    │
    ▼
cls(**data)                         ← Pydantic 校验并重建嵌套子模型
```

---

## 环境缓存策略

```python
_env_cache: ClassVar[dict[str, lmdb.Environment]] = {}
```

`ClassVar` 对 Pydantic 不可见（不参与字段校验），被所有子类共享。首次访问某路径时调用 `lmdb.open()`，后续复用已打开的 `Environment` 对象。

| 方法 | 用途 |
|---|---|
| `close_env(path)` | 关闭并逐出单个 env（换文件时使用） |
| `close_all_envs()` | 进程退出前清理全部 env，释放文件锁 |

---

## 继承与嵌套模型

只有**顶层**结构需要继承 `LMDBMixin`；内层嵌套 Pydantic 模型无需修改：

```python
class EmbeddingResult(BaseModel):      # 普通 BaseModel，无需 Mixin
    label: str
    vector: Any

class ExperimentRecord(LMDBMixin):     # 顶层，继承 Mixin
    name: str
    result: EmbeddingResult
```

`model_dump()` 已将嵌套模型递归展平为 dict；`cls(**data)` 还原时 Pydantic 自动将对应的嵌套 dict 重新构造为子模型实例。

---

## 使用示例

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

# 存储
record = ExperimentRecord(
    name="run_42",
    tags=["nlp", "encoder"],
    result=EmbeddingResult(label="bert", score=0.987,
                           vector=np.random.randn(768).astype("f4")),
    weight_matrix=np.eye(4),
    config={"lr": 1e-4, "epochs": 10},
)
key = record.store("/data/runs.lmdb")          # 返回 UUID4 字符串

# 还原（只需类型定义 + 路径）
restored = ExperimentRecord.load(key, "/data/runs.lmdb")

# 进程退出时清理
ExperimentRecord.close_all_envs()
```

---

## 限制与已知边界

| 项目 | 说明 |
|---|---|
| tuple 退化 | JSON 无 tuple 类型，tuple 字段存储后还原为 list；如需区分，建议在模型中显式使用 list |
| 级联删除 | `store()` 不提供 `delete()`；如需删除，需先 `load()` 收集所有 `__lmdb_ref__` key 再批量删除 |
| LMDB map_size | 默认 10 GiB（`_DEFAULT_MAP_SIZE`），超出需重新指定；map_size 是虚拟内存映射，不预占磁盘空间 |
| 并发写 | LMDB 同一时刻只允许一个写事务；多进程写入需在应用层加锁或分库 |
| pickle 安全性 | pickle 反序列化存在代码执行风险，不应 load 来源不可信的 LMDB 文件 |

---

## 可扩展方向

- **压缩**：在 `_put_ndarray` 中引入 `zstd`/`lz4` 压缩，对 float32 embedding 通常有 2–4× 空间收益。
- **torch.Tensor 专用路径**：仿照 ndarray 路径，直接存储 tensor 的 `__array__()` 字节，避免 pickle 开销。
- **版本字段**：在 JSON envelope 中加入 `__schema_version__`，配合 Pydantic `model_validator` 实现向前兼容的迁移。
- **多 DB 隔离**：向 `lmdb.open()` 传入 `max_dbs>0`，按模型类名创建独立 named database，避免不同模型的 blob key 命名空间冲突。

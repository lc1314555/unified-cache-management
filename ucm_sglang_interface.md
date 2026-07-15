# UCM 对接 SGLang 接口文档

## 1. 接入目标

UCM 作为 SGLang HiCache 的 L3 KV Cache 后端存储使用，负责将 SGLang Host KV Cache 中的 page 数据写入外部存储，并在后续请求中按 page key 查询、预取和恢复。

整体链路：

```text
SGLang HiCacheController
  -> UnifiedCacheStore
  -> SglangUcmConnector
  -> UcmConnectorFactoryV1
  -> UcmKVStoreBaseV1 实现
  -> UCM 底层存储后端
```

核心文件：

```text
ucm/integration/sglang/unifiedcache_store.py
ucm/integration/sglang/ucm_connector.py
ucm/store/factory_v1.py
ucm/store/ucmstore_v1.py
```

## 2. SGLang 暴露给 UCM 的接口

UCM 适配层实现 SGLang 的 `HiCacheStorage` 接口，类名：

```python
UnifiedCacheStore
```

位置：

```text
ucm/integration/sglang/unifiedcache_store.py
```

主要接口如下：

```python
class UnifiedCacheStore(HiCacheStorage):
    def __init__(
        self,
        storage_config: Optional[HiCacheStorageConfig] = None,
        context: Optional[Any] = None,
    )

    def register_mem_pool_host(self, mem_pool_host: HostKVCache)

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]

    def exists(self, key: str) -> bool

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int

    def close(self) -> None
```

## 3. 接口语义

### register_mem_pool_host

注册 SGLang 的 Host KV Cache 内存池。

```python
def register_mem_pool_host(self, mem_pool_host: HostKVCache)
```

要求：

```text
mem_pool_host.layout == "page_first"
```

否则抛出异常。UCM 当前只支持 `page_first` 布局，因为后续零拷贝读写依赖 `get_page_buffer_meta()` 获取页级内存指针。

### batch_get_v1

从 UCM 后端加载 KV Cache page 到 SGLang Host KV Cache。

```python
def batch_get_v1(
    keys: List[str],
    host_indices: torch.Tensor,
    extra_info: Optional[HiCacheStorageExtraInfo] = None,
) -> List[bool]
```

参数：

```text
keys:
  SGLang 生成的逻辑 page key 列表。

host_indices:
  Host KV Cache 中目标 page 对应的 token/page index。
  长度应为 len(keys) * page_size。

extra_info:
  SGLang 额外信息，当前 UCM 适配层不依赖该字段。
```

返回：

```text
List[bool]
  每个 key 对应一次加载结果。
  True 表示成功，False 表示失败。
```

内部流程：

```text
1. logical key -> physical key
2. physical key -> md5 bytes block_id
3. host_indices -> page buffer pointer list
4. 调用 UCM store.load_data(...)
5. 调用 UCM store.wait(...)
```

### batch_set_v1

将 SGLang Host KV Cache 中的 page 写入 UCM 后端。

```python
def batch_set_v1(
    keys: List[str],
    host_indices: torch.Tensor,
    extra_info: Optional[HiCacheStorageExtraInfo] = None,
) -> List[bool]
```

参数同 `batch_get_v1`。

返回：

```text
List[bool]
  每个 key 对应一次写入结果。
```

内部流程：

```text
1. logical key -> physical key
2. physical key -> md5 bytes block_id
3. host_indices -> page buffer pointer list
4. 调用 UCM store.dump_data(...)
5. 调用 UCM store.wait(...)
```

### exists

判断单个 key 是否存在。

```python
def exists(self, key: str) -> bool
```

内部调用：

```python
store.lookup([encoded_key])
```

MLA 模型下，如果 `tp_rank != 0`，直接返回 `True`，因为 MLA / rank-replicated 场景只需要 rank 0 真实写入和查询。

### batch_exists

查询从第一个 key 开始连续命中的 page 数。

```python
def batch_exists(
    keys: List[str],
    extra_info: Optional[HiCacheStorageExtraInfo] = None,
) -> int
```

返回：

```text
int
  从 keys[0] 开始连续存在的 key 数量。
```

内部调用：

```python
store.lookup_on_prefix(encoded_keys) + 1
```

如果一个都不存在，底层返回 `-1`，因此最终返回 `0`。

## 4. Key 编码规则

SGLang 传入的是逻辑 key，UCM 不直接使用原始 key，而是先拼接模型和并行信息生成物理 key。

非 MLA 模型：

```text
physical_key = logical_key + "_" + model_name + "_" + tp_rank + "_" + tp_size
```

MLA / rank-replicated 模型：

```text
physical_key = logical_key + "_" + model_name
```

然后编码为：

```python
block_id = hashlib.md5(physical_key.encode("utf-8")).digest()
```

最终传给 UCM store 的 key 类型是：

```python
bytes
```

## 5. 数据指针组织

UCM 使用零拷贝接口，不通过 Python tensor 返回 KV 数据，而是直接拿 Host KV Cache 的底层指针。

关键逻辑：

```python
ptr_list, _ = mem_pool_host.get_page_buffer_meta(host_indices)
```

非 MLA 模型：

```text
每个 page 对应两个指针：

[
  [k_ptr, v_ptr],
  [k_ptr, v_ptr],
  ...
]
```

MLA 模型：

```text
每个 page 对应一个指针：

[
  [kv_ptr],
  [kv_ptr],
  ...
]
```

`shard_index_list` 当前固定为：

```python
[0] * len(keys)
```

最终传入 UCM：

```python
store.load_data(key_list, shard_index_list, ptr_list)
store.dump_data(key_list, shard_index_list, ptr_list)
```

## 6. UCM Store V1 接口

UCM 底层 store 需要继承：

```python
UcmKVStoreBaseV1
```

位置：

```text
ucm/store/ucmstore_v1.py
```

核心接口：

```python
class UcmKVStoreBaseV1(ABC):
    def lookup(self, block_ids: List[bytes]) -> List[bool]

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int

    def load_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]] | np.ndarray,
    ) -> Task

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]] | np.ndarray,
        prerequisite_handle: int = 0,
    ) -> Task

    def wait(self, task: Task) -> None

    def check(self, task: Task) -> bool

    def register_memory(self, base_addr: int, total_size: int) -> None
```

SGLang 适配层实际使用：

```text
lookup
lookup_on_prefix
load_data
dump_data
wait
```

## 7. Connector 创建机制

UCM 通过工厂创建具体 store：

```python
UcmConnectorFactoryV1.create_connector(
    connector_name,
    config,
    module_path=None,
)
```

默认注册：

```text
UcmNfsStore
  -> ucm.store.pcstore.pcstore_connector_v1.UcmPcStoreV1

UcmPipelineStore
  -> ucm.store.pipeline.connector.UcmPipelineStore
```

如果传入 `module_path`，则动态导入：

```python
module = importlib.import_module(module_path)
connector_cls = getattr(module, connector_name)
```

## 8. 配置格式

UCM 适配层从 SGLang 的 `storage_config.extra_config` 读取：

```yaml
kv_connector_extra_config:
  ucm_connector_name: UcmPipelineStore
  ucm_connector_module_path: null
  ucm_connector_config:
    storage_backends: /mnt/ucm_cache
```

也可以通过环境变量指定 YAML：

```bash
UNIFIEDCACHE_CONFIG_FILE=/path/to/unifiedcache.yaml
```

配置字段说明：

```text
kv_connector_extra_config:
  UCM 对接 SGLang 的总配置节点。

ucm_connector_name:
  UCM connector 名称，例如 UcmPipelineStore。

ucm_connector_module_path:
  可选。自定义 connector 所在 Python module。

ucm_connector_config:
  传给 UCM store 的原始配置。

storage_backends:
  后端存储路径。当前代码会按 ":" 拆分为列表。
```

适配层会自动补充：

```python
store_pipeline = "Posix"
device_id = local_rank
tensor_size = 每个 tensor 分片大小
shard_size = 每个 page block 大小
block_size = 每个 page block 大小
stream_number = 8
```

## 9. SGLang 启动接入方式

当前 UCM 仓库提供了 SGLang 0.5.5 patch：

```text
ucm/integration/sglang/patch/0.5.5/sglang-adapt.patch
```

patch 主要做三件事：

```text
1. 在 SGLang StorageBackendFactory 注册 unifiedcache backend
2. 将 unifiedcache 加入 --hicache-storage-backend 参数 choices
3. 将 unifiedcache 加入 zero-copy backend 列表
```

应用 patch 后可使用：

```bash
--hicache-storage-backend unifiedcache
--hicache-mem-layout page_first
```

如果不修改 SGLang 内置 backend，也可以走 dynamic backend，但需要确保 SGLang 侧设置 `interface_v1: 1`，让 HiCacheController 使用 `batch_get_v1/batch_set_v1` 零拷贝路径。

## 10. 当前限制

```text
1. 仅支持 page_first Host KV Cache layout。
2. 仅支持 batch_get_v1 / batch_set_v1 零拷贝接口。
3. 不支持 get / set / batch_get / batch_set 旧 tensor 接口。
4. clear() 未实际清理底层 UCM 数据。
5. batch_exists 返回连续 prefix 命中数，不返回逐 key 命中列表。
6. MLA 非 tp_rank 0 查询会直接视为命中。
7. 当前适配代码默认强制 store_pipeline = "Posix"。
```

## 11. 典型读写流程

写入流程：

```text
SGLang backup/write_storage
  -> _page_set_zero_copy
  -> UnifiedCacheStore.batch_set_v1
  -> SglangUcmConnector.batch_set_v1
  -> mem_pool_host.get_page_buffer_meta
  -> UCM store.dump_data
  -> UCM store.wait
```

读取流程：

```text
SGLang prefetch
  -> batch_exists 查询 prefix 命中
  -> _page_get_zero_copy
  -> UnifiedCacheStore.batch_get_v1
  -> SglangUcmConnector.batch_get_v1
  -> mem_pool_host.get_page_buffer_meta
  -> UCM store.load_data
  -> UCM store.wait
```

命中查询流程：

```text
SGLang _storage_hit_query
  -> UnifiedCacheStore.batch_exists
  -> SglangUcmConnector.batch_exists
  -> UCM store.lookup_on_prefix
  -> 返回连续命中 page 数
```

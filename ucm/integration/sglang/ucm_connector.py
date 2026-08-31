import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import yaml
from sglang.srt.distributed.parallel_state import get_world_group

from ucm.store.factory_v1 import UcmConnectorFactoryV1

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import (
        HiCacheStorageConfig,
        HiCacheStorageExtraInfo,
    )
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


def _page_first_kv_split_buffers(mem_pool_host: "HostKVCache") -> List[torch.Tensor]:
    buffers = [mem_pool_host.k_buffer, mem_pool_host.v_buffer]
    for name, buffer in zip(("k", "v"), buffers):
        if not buffer.is_contiguous():
            raise ValueError(
                f"page_first_kv_split {name}_buffer must be contiguous"
            )
    return buffers


def _page_first_kv_split_tensor_sizes(
    mem_pool_host: "HostKVCache",
) -> List[int]:
    page_num = int(mem_pool_host.page_num)
    if page_num <= 0:
        raise ValueError(f"invalid page_num for page_first_kv_split: {page_num}")

    sizes = []
    for buffer in _page_first_kv_split_buffers(mem_pool_host):
        nbytes = int(buffer.numel()) * int(buffer.element_size())
        if nbytes % page_num != 0:
            raise ValueError(
                f"buffer bytes {nbytes} are not divisible by page_num {page_num}"
            )
        sizes.append(nbytes // page_num)
    return sizes


def _normalize_storage_backends(storage_backends: Any) -> List[str]:
    if isinstance(storage_backends, str):
        return [path for path in storage_backends.split(":") if path]
    return list(storage_backends)


def _component_storage_backends(
    storage_backends: List[str], component: str
) -> List[str]:
    component_backends = []
    for storage_backend in storage_backends:
        component_backend = Path(storage_backend) / component
        component_backend.mkdir(parents=True, exist_ok=True)
        component_backends.append(str(component_backend))
    return component_backends


def _load_extra_config_from_yaml_env() -> Optional[Dict[str, Any]]:
    cfg_path = os.environ.get("UNIFIEDCACHE_CONFIG_FILE")
    if not cfg_path:
        return None

    p = Path(cfg_path)
    if not p.is_file():
        return None

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"UNIFIEDCACHE_CONFIG_FILE YAML root must be a dict, got {type(data)}"
        )
    return data


@dataclass
class UnifiedCacheStoreConfig:
    module_path: str
    name: str
    config: Dict[str, Any]
    component_configs: Optional[Dict[str, Dict[str, Any]]] = None

    @staticmethod
    def load_from_config(
        storage_config: "HiCacheStorageConfig", mem_pool_host: "HostKVCache"
    ) -> "UnifiedCacheStoreConfig":
        extra = dict(getattr(storage_config, "extra_config", None) or {})
        if "kv_connector_extra_config" not in extra:
            yaml_extra = _load_extra_config_from_yaml_env()
            if yaml_extra is not None:
                extra.update(yaml_extra)
        if not extra:
            raise ValueError(
                "Missing extra_config: storage_config.extra_config is None and "
                "UNIFIEDCACHE_CONFIG_FILE is not set or cannot be loaded"
            )

        kvc = extra.get("kv_connector_extra_config")
        if kvc is None:
            raise ValueError(
                "Missing config: extra_config['kv_connector_extra_config']"
            )

        page_size = mem_pool_host.page_size
        page_bytes = page_size * mem_pool_host.get_size_per_token()
        tensor_size = page_bytes if storage_config.is_mla_model else page_bytes // 2
        block_size = tensor_size * (1 if storage_config.is_mla_model else 2)
        is_kv_split = (
            storage_config.is_mla_model
            and mem_pool_host.layout == "page_first_kv_split"
        )

        ucm_cfg = kvc.get("ucm_connector_config")
        connector_name = kvc.get("ucm_connector_name")
        module_path = kvc.get("ucm_connector_module_path")
        if ucm_cfg is None:
            raise ValueError(
                "Missing config: kv_connector_extra_config['ucm_connector_config']"
            )
        if connector_name is None:
            raise ValueError(
                "Missing config: kv_connector_extra_config['ucm_connector_name']"
            )

        cfg = dict(ucm_cfg)
        cfg["store_pipeline"] = "Posix"
        cfg["storage_backends"] = _normalize_storage_backends(
            cfg["storage_backends"]
        )
        cfg["device_id"] = get_world_group().local_rank
        cfg["stream_number"] = 8

        if is_kv_split:
            tensor_sizes = _page_first_kv_split_tensor_sizes(mem_pool_host)
            io_direct = bool(cfg.get("io_direct", True))
            if io_direct:
                for component, buffer, size in zip(
                    ("k", "v"),
                    _page_first_kv_split_buffers(mem_pool_host),
                    tensor_sizes,
                ):
                    if buffer.data_ptr() % 4096 or size % 4096:
                        raise ValueError(
                            f"page_first_kv_split {component} is not "
                            "4096-byte aligned: "
                            f"addr_align={buffer.data_ptr() % 4096}, "
                            f"size_align={size % 4096}; set io_direct=false"
                        )

            component_configs = {}
            for component, size in zip(("k", "v"), tensor_sizes):
                component_cfg = dict(cfg)
                component_cfg.pop("tensor_size_list", None)
                component_cfg["storage_backends"] = _component_storage_backends(
                    cfg["storage_backends"], component
                )
                component_cfg["tensor_size"] = size
                component_cfg["shard_size"] = size
                component_cfg["block_size"] = size
                component_configs[component] = component_cfg

            return UnifiedCacheStoreConfig(
                module_path=module_path,
                name=connector_name,
                config=cfg,
                component_configs=component_configs,
            )

        cfg["tensor_size"] = tensor_size
        cfg["shard_size"] = block_size
        cfg["block_size"] = block_size

        return UnifiedCacheStoreConfig(
            module_path=module_path, name=connector_name, config=cfg
        )


class SglangUcmConnector:
    def __init__(
        self,
        store,
        mem_pool_host: "HostKVCache",
        storage_config: "HiCacheStorageConfig",
        storage_backends: List[str],
        v_store=None,
    ):
        self.store = store
        self.k_store = store
        self.v_store = v_store
        self.mem_pool_host = mem_pool_host
        self.storage_backends = storage_backends

        self.dtype = mem_pool_host.dtype
        self.page_size = mem_pool_host.page_size
        self.model = storage_config.model_name
        self.is_mla = storage_config.is_mla_model
        self.tp_rank = storage_config.tp_rank
        self.tp_size = storage_config.tp_size
        self.is_kv_split = (
            self.is_mla and mem_pool_host.layout == "page_first_kv_split"
        )
        self.cache_nums = 2 if self.is_kv_split or not self.is_mla else 1
        self.split_buffers = (
            _page_first_kv_split_buffers(mem_pool_host)
            if self.is_kv_split
            else []
        )
        self.split_tensor_sizes = (
            _page_first_kv_split_tensor_sizes(mem_pool_host)
            if self.is_kv_split
            else []
        )

        self.config_suffix = self._build_config_suffix()

    @classmethod
    def from_hicache(
        cls,
        storage_config: "HiCacheStorageConfig",
        mem_pool_host: "HostKVCache",
    ) -> "SglangUcmConnector":
        if mem_pool_host is None:
            raise ValueError("mem_pool_host must be provided for UnifiedCache")
        ucm_store_config = UnifiedCacheStoreConfig.load_from_config(
            storage_config, mem_pool_host
        )
        if ucm_store_config.component_configs is not None:
            k_config = ucm_store_config.component_configs["k"]
            v_config = ucm_store_config.component_configs["v"]
            logger.info(
                "Creating split MLA Posix stores: K=%s, V=%s",
                k_config["storage_backends"],
                v_config["storage_backends"],
            )
            k_store = UcmConnectorFactoryV1.create_connector(
                ucm_store_config.name, k_config, ucm_store_config.module_path
            )
            v_store = UcmConnectorFactoryV1.create_connector(
                ucm_store_config.name, v_config, ucm_store_config.module_path
            )
            return cls(
                k_store,
                mem_pool_host,
                storage_config,
                k_config["storage_backends"],
                v_store=v_store,
            )

        store = UcmConnectorFactoryV1.create_connector(
            ucm_store_config.name,
            ucm_store_config.config,
            ucm_store_config.module_path,
        )
        return cls(
            store,
            mem_pool_host,
            storage_config,
            ucm_store_config.config["storage_backends"],
        )

    def _encode_key(self, key: str) -> bytes:
        return hashlib.md5(key.encode("utf-8")).digest()

    def _encode_keys(self, keys: List[str]) -> List[bytes]:
        return [self._encode_key(key) for key in keys]

    def _build_config_suffix(self) -> str:
        model_name = "-".join(self.model.split("/")) if self.model else ""
        if self.is_kv_split:
            tensor_fingerprint = "-".join(
                str(size) for size in self.split_tensor_sizes
            )
            return (
                f"_{model_name}_{self.mem_pool_host.layout}_p{self.page_size}_"
                f"tp{self.tp_size}_{tensor_fingerprint}"
            )
        if self.is_mla:
            return f"_{model_name}"
        return f"_{model_name}_{self.tp_rank}_{self.tp_size}"

    def _get_physical_key(self, logical_key: str) -> str:
        return logical_key + self.config_suffix

    def _get_physical_keys(self, logical_keys: List[str]) -> List[str]:
        return [self._get_physical_key(key) for key in logical_keys]

    def _get_component_physical_keys(
        self, logical_keys: List[str], component: str
    ) -> List[str]:
        return [
            f"{self._get_physical_key(key)}_{component}" for key in logical_keys
        ]

    def _generate_split_task(
        self,
        encoded_keys: List[bytes],
        host_indices: torch.Tensor,
        component_index: int,
    ):
        if not encoded_keys:
            return [], [], []

        indices = host_indices.tolist()
        if len(indices) % self.page_size != 0:
            raise ValueError("host_indices length must be a multiple of page_size")
        if len(indices) // self.page_size != len(encoded_keys):
            raise ValueError(
                "page count mismatch between keys and host_indices: "
                f"{len(encoded_keys)} != {len(indices) // self.page_size}"
            )

        buffer = self.split_buffers[component_index]
        tensor_size = self.split_tensor_sizes[component_index]
        ptr_list = []
        for offset in range(0, len(indices), self.page_size):
            first_token_index = int(indices[offset])
            if first_token_index % self.page_size != 0:
                raise ValueError(
                    "page_first_kv_split host index must start at a page boundary"
                )
            page_index = first_token_index // self.page_size
            ptr_list.append([buffer.data_ptr() + page_index * tensor_size])

        return encoded_keys, [0] * len(encoded_keys), ptr_list

    def _split_tasks(self, keys: List[str], host_indices: torch.Tensor):
        tasks = []
        for component_index, component in enumerate(("k", "v")):
            encoded_keys = self._encode_keys(
                self._get_component_physical_keys(keys, component)
            )
            tasks.append(
                self._generate_split_task(
                    encoded_keys, host_indices, component_index
                )
            )
        return tasks

    def _run_split_transfer(
        self, operation: str, keys: List[str], host_indices: torch.Tensor
    ) -> bool:
        stores = (self.k_store, self.v_store)
        component_tasks = self._split_tasks(keys, host_indices)
        submitted_tasks = []
        success = True

        for component, store, (key_list, shard_indices, ptr_list) in zip(
            ("K", "V"), stores, component_tasks
        ):
            try:
                submit = (
                    store.load_data if operation == "load" else store.dump_data
                )
                task = submit(key_list, shard_indices, ptr_list)
                submitted_tasks.append((component, store, task))
            except RuntimeError as e:
                logger.error(
                    "UnifiedCache %s MLA %s submit failed: %s",
                    operation,
                    component,
                    e,
                )
                success = False

        for component, store, task in submitted_tasks:
            try:
                store.wait(task)
            except RuntimeError as e:
                logger.error(
                    "UnifiedCache %s MLA %s wait failed: %s",
                    operation,
                    component,
                    e,
                )
                success = False

        return success

    def _generate_task(
        self,
        encoded_keys: List[bytes],
        host_indices: torch.Tensor,
    ):
        if not encoded_keys:
            return [], [], []

        shard_index_list = [0] * len(encoded_keys)
        ptr_list, _ = self.mem_pool_host.get_page_buffer_meta(host_indices)

        if not self.is_mla:
            ptr_list = [list(p) for p in zip(ptr_list[::2], ptr_list[1::2])]
        else:
            ptr_list = [[p] for p in ptr_list]

        return encoded_keys, shard_index_list, ptr_list

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        if not keys:
            return []

        if self.is_kv_split:
            success = self._run_split_transfer("load", keys, host_indices)
            return [success] * len(keys)

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        key_list, shard_index_list, ptr_list = self._generate_task(
            encoded_keys, host_indices
        )

        task = self.store.load_data(key_list, shard_index_list, ptr_list)
        try:
            self.store.wait(task)
        except RuntimeError as e:
            logger.error(f"UnifiedCache load KVCache failed: {e}")
            return [False] * len(keys)

        return [True] * len(keys)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        if not keys:
            return []

        if self.is_kv_split:
            success = self._run_split_transfer("dump", keys, host_indices)
            return [success] * len(keys)

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        key_list, shard_index_list, ptr_list = self._generate_task(
            encoded_keys, host_indices
        )

        task = self.store.dump_data(key_list, shard_index_list, ptr_list)
        try:
            self.store.wait(task)
        except RuntimeError as e:
            logger.error(f"UnifiedCache dump KVCache failed: {e}")
            return [False] * len(keys)

        return [True] * len(keys)

    def exists(self, key: str) -> bool:
        if self.is_mla and self.tp_rank != 0:
            return True

        if self.is_kv_split:
            results = []
            for component, store in zip(("k", "v"), (self.k_store, self.v_store)):
                encoded_key = self._encode_keys(
                    self._get_component_physical_keys([key], component)
                )
                results.append(store.lookup(encoded_key)[0] == 1)
            return all(results)

        result = self.store.lookup(self._encode_keys([self._get_physical_key(key)]))
        return result[0] == 1

    def batch_exists(
        self, keys: List[str], extra_info: Optional["HiCacheStorageExtraInfo"] = None
    ) -> int:
        if not keys:
            return 0
        if self.is_mla and self.tp_rank != 0:
            return len(keys)

        if self.is_kv_split:
            prefixes = []
            for component, store in zip(("k", "v"), (self.k_store, self.v_store)):
                encoded_keys = self._encode_keys(
                    self._get_component_physical_keys(keys, component)
                )
                prefixes.append(store.lookup_on_prefix(encoded_keys))
            return min(prefixes) + 1

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        return self.store.lookup_on_prefix(encoded_keys) + 1

    def close(self) -> None:
        seen = set()
        for store in (self.k_store, self.v_store):
            if store is None or id(store) in seen:
                continue
            seen.add(id(store))
            close = getattr(store, "close", None)
            if callable(close):
                close()

    def get_stats(self):
        return None

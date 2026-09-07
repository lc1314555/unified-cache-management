import sys
import types
from types import SimpleNamespace


def _install_sglang_stubs():
    sglang = sys.modules.setdefault("sglang", types.ModuleType("sglang"))
    srt = sys.modules.setdefault("sglang.srt", types.ModuleType("sglang.srt"))
    distributed = sys.modules.setdefault(
        "sglang.srt.distributed", types.ModuleType("sglang.srt.distributed")
    )
    parallel_state = types.ModuleType("sglang.srt.distributed.parallel_state")
    parallel_state.get_world_group = lambda: SimpleNamespace(local_rank=0)
    sys.modules.setdefault(
        "sglang.srt.distributed.parallel_state", parallel_state
    )
    sglang.srt = srt
    srt.distributed = distributed


_install_sglang_stubs()

from ucm.integration.sglang.ucm_connector import (  # noqa: E402
    SglangUcmConnector,
    UnifiedCacheStoreConfig,
    resolve_v1_host_pool,
)


class FakeBuffer:
    def __init__(self, address, nbytes):
        self.address = address
        self.nbytes = nbytes

    def is_contiguous(self):
        return True

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1

    def data_ptr(self):
        return self.address


class FakeHostIndices:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeStore:
    def __init__(self):
        self.loads = []
        self.dumps = []
        self.waits = []
        self.lookup_result = [True]
        self.prefix = 0

    def load_data(self, keys, indexes, pointers):
        self.loads.append((keys, indexes, pointers))
        return ("load", len(self.loads))

    def dump_data(self, keys, indexes, pointers):
        self.dumps.append((keys, indexes, pointers))
        return ("dump", len(self.dumps))

    def wait(self, task):
        self.waits.append(task)

    def lookup(self, keys):
        return self.lookup_result

    def lookup_on_prefix(self, keys):
        return self.prefix


def _make_pool(with_indexer=False):
    pool = SimpleNamespace(
        page_size=2,
        page_num=4,
        layout="page_first_kv_split",
        dtype="fake",
        k_buffer=FakeBuffer(0x100000, 400),
        v_buffer=FakeBuffer(0x200000, 200),
        get_size_per_token=lambda: 75,
    )
    pool.index_k_buffer = FakeBuffer(0x300000, 120) if with_indexer else None
    return pool


def _make_storage_config(storage_backends="/mnt/ucm0:/mnt/ucm1"):
    return SimpleNamespace(
        is_mla_model=True,
        model_name="deepseek-v2-lite",
        tp_rank=0,
        tp_size=1,
        extra_config={
            "kv_connector_extra_config": {
                "ucm_connector_name": "UcmPipelineStore",
                "ucm_connector_config": {
                    "storage_backends": storage_backends,
                    "io_direct": False,
                },
            }
        },
    )


def test_split_mla_builds_independent_k_and_v_posix_configs(tmp_path):
    backend0 = tmp_path / "ucm0"
    backend1 = tmp_path / "ucm1"
    config = UnifiedCacheStoreConfig.load_from_config(
        _make_storage_config(f"{backend0}:{backend1}"), _make_pool()
    )

    assert config.component_configs is not None
    k_config = config.component_configs["k"]
    v_config = config.component_configs["v"]
    assert k_config["store_pipeline"] == "Posix"
    assert v_config["store_pipeline"] == "Posix"
    assert k_config["storage_backends"] == [str(backend0 / "k"), str(backend1 / "k")]
    assert v_config["storage_backends"] == [str(backend0 / "v"), str(backend1 / "v")]
    assert all((backend / "k").is_dir() for backend in (backend0, backend1))
    assert all((backend / "v").is_dir() for backend in (backend0, backend1))
    assert (k_config["tensor_size"], k_config["shard_size"]) == (100, 100)
    assert (v_config["tensor_size"], v_config["shard_size"]) == (50, 50)


def test_split_mla_submits_k_and_v_to_different_stores():
    k_store = FakeStore()
    v_store = FakeStore()
    connector = SglangUcmConnector(
        k_store,
        _make_pool(),
        _make_storage_config(),
        ["/mnt/ucm0/k"],
        v_store=v_store,
    )
    host_indices = FakeHostIndices([0, 1, 2, 3])

    assert connector.batch_set_v1(["page0", "page1"], host_indices) == [True, True]
    assert k_store.dumps[0][1:] == ([0, 0], [[0x100000], [0x100000 + 100]])
    assert v_store.dumps[0][1:] == ([0, 0], [[0x200000], [0x200000 + 50]])
    assert len(k_store.waits) == 1
    assert len(v_store.waits) == 1
    assert k_store.dumps[0][0] != v_store.dumps[0][0]


def test_split_mla_requires_both_stores_for_a_hit():
    k_store = FakeStore()
    v_store = FakeStore()
    connector = SglangUcmConnector(
        k_store,
        _make_pool(),
        _make_storage_config(),
        ["/mnt/ucm0/k"],
        v_store=v_store,
    )

    k_store.lookup_result = [True]
    v_store.lookup_result = [False]
    assert connector.exists("page0") is False

    k_store.prefix = 4
    v_store.prefix = 2
    assert connector.batch_exists(["0", "1", "2", "3", "4"]) == 3


def test_split_ascend_mla_adds_optional_indexer_store(tmp_path):
    backend = tmp_path / "ucm"
    pool = _make_pool(with_indexer=True)
    config = UnifiedCacheStoreConfig.load_from_config(
        _make_storage_config(str(backend)), pool
    )

    assert list(config.component_configs) == ["k", "v", "indexer"]
    indexer_config = config.component_configs["indexer"]
    assert indexer_config["storage_backends"] == [str(backend / "indexer")]
    assert indexer_config["tensor_size"] == 30
    assert indexer_config["shard_size"] == 30


def test_split_ascend_mla_transfers_and_requires_indexer():
    stores = {name: FakeStore() for name in ("k", "v", "indexer")}
    connector = SglangUcmConnector(
        stores["k"],
        _make_pool(with_indexer=True),
        _make_storage_config(),
        ["/mnt/ucm0/k"],
        v_store=stores["v"],
        component_stores=stores,
    )
    host_indices = FakeHostIndices([0, 1, 2, 3])

    assert connector.batch_set_v1(["page0", "page1"], host_indices) == [True, True]
    assert stores["k"].dumps[0][1:] == (
        [0, 0],
        [[0x100000], [0x100000 + 100]],
    )
    assert stores["v"].dumps[0][1:] == (
        [0, 0],
        [[0x200000], [0x200000 + 50]],
    )
    assert stores["indexer"].dumps[0][1:] == (
        [0, 0],
        [[0x300000], [0x300000 + 30]],
    )

    assert connector.batch_get_v1(["page0", "page1"], host_indices) == [True, True]
    assert stores["indexer"].loads[0][1:] == (
        [0, 0],
        [[0x300000], [0x300000 + 30]],
    )

    stores["k"].lookup_result = [True]
    stores["v"].lookup_result = [True]
    stores["indexer"].lookup_result = [False]
    assert connector.exists("page0") is False

    stores["k"].prefix = 4
    stores["v"].prefix = 4
    stores["indexer"].prefix = 1
    assert connector.batch_exists(["0", "1", "2", "3", "4"]) == 2


def test_host_pool_group_resolves_to_v1_anchor_pool(tmp_path):
    pool = _make_pool(with_indexer=True)
    group = SimpleNamespace(anchor_entry=SimpleNamespace(host_pool=pool))

    assert resolve_v1_host_pool(group) is pool
    config = UnifiedCacheStoreConfig.load_from_config(
        _make_storage_config(str(tmp_path / "ucm")), group
    )
    assert list(config.component_configs) == ["k", "v", "indexer"]

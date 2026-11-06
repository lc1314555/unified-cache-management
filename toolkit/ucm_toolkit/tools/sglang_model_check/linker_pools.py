"""Reuse SGLang's device-pool dispatcher, request pool and allocator."""

from copy import copy
from types import SimpleNamespace

from .native_probe import _dsv4_main_layer_ratios, cache_builder, native_family
from .runner import _native_cache_dtype


def build_linker_pools(config, model, requirements, environment, layer_range=None):
    from sglang.srt.configs.hybrid_arch import (
        hybrid_gdn_config, hybrid_kda_config, mambaish_config,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import (
        _PoolSizes, _get_dsv4_compress_state_dtypes,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import resolve_compress_state_ring_size

    family = native_family(requirements)
    model = copy(model)
    pages = config.pages + 2
    size = pages * config.page_size
    model.context_len = size
    if family == "deepseek_v4":
        model.compress_ratios = _dsv4_main_layer_ratios(model)
    start, end = layer_range or (0, model.num_hidden_layers)
    dtype = _native_cache_dtype(model, config, family, environment["platform"]) if family else model.dtype
    builder = cache_builder(model, config, dtype, start, end, is_mla=requirements.is_mla)
    builder.device = environment["device_name"]
    builder.mambaish_config = mambaish_config(model)
    builder.hybrid_gdn_config = hybrid_gdn_config(model)
    builder.hybrid_kda_config = hybrid_kda_config(model)
    builder.spec_algorithm = SpeculativeAlgorithm.NONE
    builder.is_hybrid_swa = bool(getattr(model, "sliding_window_size", None))
    builder.sliding_window_size = getattr(model, "sliding_window_size", None)
    builder.draft_parallel_layout = None
    builder.draft_model_idx = None
    builder.draft_owns_its_depth = False
    builder.draft_swa_full_capacity = False
    builder.modelslim_cache_layout = None
    builder.kv_shard_rank = None
    builder.kv_shard_size = 1
    builder.kv_shard_spec = None
    builder.kv_shard_group = None
    builder._unified_plan = None
    builder._unified_sized = None
    builder.token_to_kv_pool_allocator = None
    builder.model = SimpleNamespace(quant_config=None)
    builder.spec_aux_config = SimpleNamespace()
    request_pool = builder._build_req_to_token_pool(max_num_reqs=2)
    state = request_pool.state
    c4_dtype, c128_dtype = _get_dsv4_compress_state_dtypes()
    sizes = _PoolSizes(
        max_total_num_tokens=size, max_running_requests=2,
        full_max_total_num_tokens=size, swa_max_total_num_tokens=size,
        c4_max_total_num_tokens=max(size // 4, config.page_size),
        c128_max_total_num_tokens=max(size // 128, config.c128_page_size),
        c4_state_pool_size=pages * resolve_compress_state_ring_size(4),
        c128_state_pool_size=pages * resolve_compress_state_ring_size(128),
        c4_state_dtype=c4_dtype, c128_state_dtype=c128_dtype,
    )
    pool = builder._build_token_to_kv_pool(
        sizes=sizes, is_dsa_model=family == "dsa",
        is_dsv4_model=family == "deepseek_v4", req_to_token_pool=request_pool,
        state=state,
    )
    allocator = builder._build_token_to_kv_pool_allocator(
        sizes=sizes, token_to_kv_pool=pool, is_dsv4_model=family == "deepseek_v4",
        req_to_token_pool=request_pool, token_to_kv_pool_allocator=None,
    )
    allocator.state = state
    return pool, allocator, request_pool

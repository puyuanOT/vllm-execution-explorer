"""CPU-only checks of retained GPU snapshots; never reads live device memory."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import torch

ROOT = Path('/home/ubuntu/vllm-execution-explorer/docs/qualification/demo-expansion')

def load(name):
    root = ROOT / f'{name}-qualified' / 'six-request' / 'recording'
    manifest = json.loads((root / 'manifest.json').read_text())
    source = root / manifest['profiling']['normalized_artifact']['path']
    encoded = source.read_bytes()
    return manifest, json.loads(encoded), hashlib.sha256(encoded).hexdigest()

baseline, bf16, baseline_hash = load('bf16-final')
manifest, profile, profile_hash = load('fp8-kv')
snapshots = profile['tensor_snapshots']
expected = set(itertools.product([0, 1], [0, 18, 35], [0, 1], ['K', 'V']))
actual = {(s['token_position'], s['layer_index'], s['kv_head'], s['kind']) for s in snapshots}
assert actual == expected and len(snapshots) == 24
writes = {w['operation_id']: w for w in profile['cache_writes']}
layers = {l['layer_index']: l for l in manifest['kv_cache']['layers']}
checks = []
for snapshot in snapshots:
    layer = layers[snapshot['layer_index']]
    write = writes[snapshot['operation_id']]
    participant = next(p for p in write['participants'] if p['request_id'] == snapshot['request_id'] and p['token_position'] == snapshot['token_position'])
    for key in ['pool', 'cache_group', 'block_id', 'allocation_generation', 'slot']:
        assert participant[key] == snapshot[key]
    assert snapshot['shape'] == [16] and snapshot['dimension_start'] == 0
    assert layer['dtype'] == snapshot['dtype'] == 'uint8'
    assert layer['numerical_encoding'] == snapshot['quantization']['format'] == 'fp8_e4m3'
    assert snapshot['observation_boundary'] == snapshot['quantization']['scale']['observation_boundary'] == 'same_cuda_stream_after_native_cache_update_before_reuse'
    assert participant['kernel_block_id'] == snapshot['block_id'] * layer['kernel_blocks_per_manager_block'] + snapshot['slot'] // layer['kernel_block_size']
    assert participant['kernel_slot'] == snapshot['slot'] % layer['kernel_block_size']
    assert 0 <= participant['kernel_block_id'] < layer['num_kernel_blocks']
    assert 0 <= participant['kernel_slot'] < layer['kernel_block_size']
    assert 0 <= snapshot['kv_head'] < layer['num_kv_heads']
    assert snapshot['dimension_start'] + len(snapshot['values']) <= layer['head_dim']
    raw = torch.tensor(snapshot['quantization']['raw_bits'], dtype=torch.uint8, device='cpu')
    decoded = raw.view(torch.float8_e4m3fn).float()
    scale = snapshot['quantization']['scale']['value']
    assert decoded.tolist() == snapshot['values']
    assert [value * scale for value in decoded.tolist()] == snapshot['quantization']['reconstructed_values']
    assert snapshot['byte_count'] == raw.numel() + 4
    offsets = [layer['storage_offset'] + participant['kernel_block_id']*layer['strides'][0] + snapshot['kv_head']*layer['strides'][1] + participant['kernel_slot']*layer['strides'][2] + (snapshot['dimension_start'] + dimension + (0 if snapshot['kind'] == 'K' else layer['head_dim']))*layer['strides'][3] for dimension in range(16)]
    storage = next(s for s in manifest['kv_cache']['storages'] if s['storage_id'] == layer['storage_id'])
    assert all(0 <= offset < storage['byte_count'] for offset in offsets)
    checks.append({'snapshot_id': snapshot['snapshot_id'], 'layer_index': snapshot['layer_index'], 'token_position': snapshot['token_position'], 'kv_head': snapshot['kv_head'], 'kind': snapshot['kind'], 'native_cpu_decoding_equal': True, 'scale': scale, 'address_metadata_consistent': True, 'retained_storage_byte_offsets': offsets})
assert sum(s['byte_count'] for s in snapshots) == profile['tensor_capture']['retained_bytes'] == 480
first_layer = []
for snapshot in snapshots:
    if snapshot['layer_index'] != 0:
        continue
    peer = next(s for s in bf16['tensor_snapshots'] if (s['token_position'],s['layer_index'],s['kv_head'],s['kind']) == (snapshot['token_position'],snapshot['layer_index'],snapshot['kv_head'],snapshot['kind']))
    assert peer['step_id'] == snapshot['step_id'] == 'step-1'
    assert peer['request_id'] == snapshot['request_id'] == 'request-1'
    scale = snapshot['quantization']['scale']['value']
    cast = (torch.tensor(peer['values'], dtype=torch.bfloat16, device='cpu').float() / scale).to(torch.float8_e4m3fn).view(torch.uint8).tolist()
    first_layer.append({'snapshot_id': snapshot['snapshot_id'], 'baseline_snapshot_id': peer['snapshot_id'], 'native_cpu_cast_raw_bytes_equal': cast == snapshot['quantization']['raw_bits'], 'compared_bytes': len(cast)})
assert all(check['native_cpu_cast_raw_bytes_equal'] for check in first_layer)
print(json.dumps({'scope': 'CPU-only retained artifact verification. Address metadata is checked against recorded native mapping and storage geometry; current GPU memory is not read.', 'torch': torch.__version__, 'fp8_run_id': manifest['run_id'], 'bf16_run_id': baseline['run_id'], 'fp8_profile_sha256': profile_hash, 'bf16_profile_sha256': baseline_hash, 'snapshot_count': len(snapshots), 'decoded_components': sum(len(s['values']) for s in snapshots), 'retained_bytes': profile['tensor_capture']['retained_bytes'], 'snapshot_checks': checks, 'first_layer_cross_capture': first_layer, 'first_layer_compared_bytes': sum(c['compared_bytes'] for c in first_layer), 'higher_layer_comparison': 'Not required to equal the BF16 case: FP8 attention cache can change subsequent layer inputs.', 'outcome': 'PASS'}, indent=2))

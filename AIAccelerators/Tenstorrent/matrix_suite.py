"""Read the canonical MATRIX.md; never form a Cartesian product of N and K."""
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
MATRIX = BASE.parents[1] / 'MLBench/mlops/MATRIX.md'
FIDELITIES = {'bfp8': ('lofi', 'hifi2'), 'bf16': ('hifi2', 'hifi4'), 'fp32': ('hifi4',)}


def shapes():
    family = None
    result = []
    for line in MATRIX.read_text().splitlines():
        if line.startswith('### ') and 'Square' in line:
            family = 'square'
        elif line.startswith('### ') and 'Short and wide' in line:
            family = 'short_wide'
        elif line.startswith('### ') and 'Tall and narrow' in line:
            family = 'tall_narrow'
        match = re.match(r'^\s*\|?\s*M,\s*(\d+),\s*(\d+)\s*\|', line)
        if match:
            if family is None:
                raise ValueError('Shape outside a known MATRIX.md section')
            n, k = map(int, match.groups())
            result.append((family, n, k))
    if not result or len(result) != len(set(result)):
        raise ValueError('Empty or duplicate MATRIX.md shapes')
    return result


def cases():
    for family, n, k in shapes():
        for op in ('gemm', 'gemv', 'gelu', 'softmax'):
            for m in ((1,) if op == 'gemv' else (64, 512, 2048)):
                for dtype, fidelities in FIDELITIES.items():
                    for fidelity in (fidelities if op in ('gemm', 'gemv') else ('none',)):
                        for layout in (('interleaved',) if op in ('gemm', 'gemv') else ('interleaved', 'sharded')):
                            yield dict(shape_family=family, operation=op, M=m, N=n, K=k,
                                       dtype=dtype, math_fidelity=fidelity, memory_layout=layout)


def case_id(c):
    return 'tt_matrix_' + '_'.join(str(c[k]) for k in ('shape_family', 'operation', 'M', 'N', 'K', 'dtype', 'math_fidelity', 'memory_layout'))


def generate():
    import json
    jobs = []
    for c in cases():
        matrix = c['operation'] in ('gemm', 'gemv')
        args = f"{c['M']} {c['K']} {c['N']} --op matmul --math-fidelity {c['math_fidelity']} --compute-grid-x 8 --compute-grid-y 8" if matrix else f"{c['M']} {c['N']} --op {c['operation']} --softmax-dim -1"
        command = f"bash scripts/launch_benchmark.sh {'gemm' if matrix else 'nonlinearity'} --result-id {case_id(c)} --no-power -- {args} --dtype {c['dtype']} --layout tile --memory-layout {c['memory_layout']} --shard-grid-x 8 --shard-grid-y 8 --warmup=5 --run=10"
        jobs.append({'tag': case_id(c), 'command': command})
    # JSON is valid YAML and avoids an extra generation dependency.
    (BASE/'jobs.yaml').write_text(json.dumps({'sequential': True, 'jobs': [{'config': 'tenstorrent_local', 'config_jobs': jobs}]}, indent=2)+'\n')
    print(f'Generated {len(jobs)} sequential jobs from {MATRIX}')


if __name__ == '__main__':
    generate()

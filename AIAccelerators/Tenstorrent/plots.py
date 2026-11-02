"""One combined throughput figure per MATRIX.md shape family."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matrix_suite import shapes, FIDELITIES


def plot(data, output):
    required = {'shape_family','operation','M','N','K','dtype','math_fidelity','memory_layout','throughput','throughput_unit'}
    if not required.issubset(data.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(data.columns))}')
    if not np.isfinite(data.throughput).all() or (data.throughput <= 0).any():
        raise ValueError('Throughput must be positive and finite')
    output.mkdir(parents=True, exist_ok=True)
    keys = ['shape_family','operation','M','N','K','dtype','math_fidelity','memory_layout','throughput_unit']
    summary = data.groupby(keys, dropna=False).throughput.agg(['median','count']).reset_index()
    colors = {'gemm':'#0072B2','gemv':'#D55E00','gelu':'#009E73','softmax':'#7A6A00'}
    for family in dict.fromkeys(s[0] for s in shapes()):
        pairs = [(n,k) for f,n,k in shapes() if f == family]
        fig, axes = plt.subplots(3, 3, figsize=(19, 13), squeeze=False)
        for i, dtype in enumerate(FIDELITIES):
            for j, m in enumerate((64,512,2048)):
                left = axes[i,j]
                right = left.twinx()
                handles = []
                subset = summary[(summary.shape_family == family) & (summary.dtype == dtype)]
                for op in colors:
                    view = subset[(subset.operation == op) & (subset.M == (1 if op == 'gemv' else m))]
                    for (fidelity, layout), group in view.groupby(['math_fidelity','memory_layout']):
                        values = {(int(r.N),int(r.K)):r.median for r in group.itertuples()}
                        y = [values.get(pair, float('nan')) for pair in pairs]
                        axis = left if op in ('gemm','gemv') else right
                        style = '--' if fidelity in ('hifi2','hifi4') and fidelity != FIDELITIES[dtype][0] else '-'
                        if op in ('gelu','softmax'):
                            style = ':' if layout == 'interleaved' else '-.'
                        label = f'{op.upper()} / {fidelity if op in ("gemm","gemv") else layout}'
                        handles += axis.plot(range(len(pairs)), y, marker={'gemm':'o','gemv':'s','gelu':'^','softmax':'v'}[op], color=colors[op], linestyle=style, label=label)
                left.set_xticks(range(len(pairs)), [f'{n} / {k}' for n,k in pairs], rotation=35, ha='right', fontsize=8)
                left.set_xlabel('N / K (categorical shape pairs)')
                left.set_ylabel('GEMM / GEMV (GFLOP/s)')
                right.set_ylabel('GELU / Softmax (Gelem/s)')
                left.set_yscale('log')
                right.set_yscale('log')
                left.grid(alpha=.2)
                left.set_title(f'{dtype.upper()} · M={m} (GEMV M=1)')
                if handles:
                    left.legend(handles=handles, fontsize=6, loc='best', ncol=2)
                else:
                    left.text(.5,.5,'No completed measurements', transform=left.transAxes, ha='center')
        fig.suptitle(f'Tenstorrent · {family.replace("_", " ")} · median kernel throughput\nMissing measurements leave gaps; left and right axes use different units')
        fig.tight_layout(rect=(0,0,1,.95))
        stem = output/f'tenstorrent_{family}'
        fig.savefig(stem.with_suffix('.pdf'))
        fig.savefig(stem.with_suffix('.png'), dpi=160)
        plt.close(fig)
        summary[summary.shape_family == family].to_csv(output/f'tenstorrent_{family}_data.csv', index=False)
        print(stem.with_suffix('.pdf'))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('files', type=Path, nargs='*')
    cli.add_argument('--output-dir', '--outdir', type=Path, default=Path('plots'))
    args = cli.parse_args()
    files = args.files or sorted(Path('results').glob('tenstorrent_matrix_*_data.csv'))
    if not files:
        raise SystemExit('No results/tenstorrent_matrix_<cluster>_data.csv files found; run parse_results.py first.')
    data = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    clusters = list(data.groupby('cluster', dropna=False)) if 'cluster' in data else [(None, data)]
    for cluster, group in clusters:
        if len(clusters) > 1:
            if not isinstance(cluster, str) or not cluster or Path(cluster).name != cluster or cluster in ('.', '..'):
                raise ValueError(f'Invalid cluster name: {cluster!r}')
            plot(group, args.output_dir / cluster)
        else:
            plot(group, args.output_dir)


if __name__ == '__main__':
    main()

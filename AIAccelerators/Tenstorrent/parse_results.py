"""Export canonical matrix campaign results from SbatchMan or saved stdout."""
import argparse
import json
from pathlib import Path
import pandas as pd
from parser import parse, parse_stdout, CASE_MAP


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--logs', type=Path, nargs='*', help='Saved stdout files; names must be exact job tags plus .log')
    cli.add_argument('--output', type=Path, help='Explicit CSV path for a single cluster')
    cli.add_argument('--cluster', default='offline', help='Cluster name for saved logs (default: offline)')
    args = cli.parse_args()
    rows = []
    if args.logs is not None:
        for path in args.logs:
            row = parse_stdout(path.read_text(), path.stem, {'source': str(path), 'cluster': args.cluster})
            if row:
                rows.append(row)
    else:
        import sbatchman as sbm
        for job in sbm.jobs_list(status=[sbm.Status.COMPLETED], from_active=True, from_archived=True):
            parsed = parse(job)
            if parsed:
                rows.append(parsed['tenstorrent'])
    if not rows:
        raise SystemExit('No completed MATRIX.md results found.')
    frame = pd.DataFrame([{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in r.items()} for r in rows])
    groups = list(frame.groupby('cluster', dropna=False))
    if args.output is not None and len(groups) != 1:
        raise SystemExit('--output requires results from exactly one cluster.')
    for cluster, group in groups:
        if not isinstance(cluster, str) or not cluster or Path(cluster).name != cluster or cluster in ('.', '..'):
            raise ValueError(f'Invalid cluster name: {cluster!r}')
        output = args.output or Path('results') / f'tenstorrent_matrix_{cluster}_data.csv'
        output.parent.mkdir(parents=True, exist_ok=True)
        group.sort_values(['shape_family','operation','M','N','K','dtype']).to_csv(output, index=False)
        missing = sorted(set(CASE_MAP)-set(group.tag))
        output.with_suffix('.coverage.json').write_text(json.dumps({'cluster': cluster, 'expected': len(CASE_MAP), 'completed': len(set(group.tag)), 'missing_tags': missing}, indent=2)+'\n')
        print(f'Wrote {len(group)} runs to {output}; {len(missing)} configurations missing (see coverage JSON).')



if __name__ == '__main__':
    main()

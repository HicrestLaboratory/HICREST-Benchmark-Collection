"""Export canonical matrix campaign results from SbatchMan or saved stdout."""
import argparse
import json
from pathlib import Path
import pandas as pd
from parser import parse, parse_stdout, CASE_MAP


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--logs', type=Path, nargs='*', help='Saved stdout files; names must be exact job tags plus .log')
    cli.add_argument('--output', type=Path, default=Path('results/tenstorrent_matrix_data.csv'))
    args = cli.parse_args()
    rows = []
    if args.logs is not None:
        for path in args.logs:
            row = parse_stdout(path.read_text(), path.stem, {'source': str(path)})
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(['shape_family','operation','M','N','K','dtype']).to_csv(args.output, index=False)
    missing = sorted(set(CASE_MAP)-set(frame.tag))
    args.output.with_suffix('.coverage.json').write_text(json.dumps({'expected': len(CASE_MAP), 'completed': len(set(frame.tag)), 'missing_tags': missing}, indent=2)+'\n')
    print(f'Wrote {len(frame)} runs; {len(missing)} configurations missing (see coverage JSON).')


if __name__ == '__main__':
    main()

from pprint import pprint
import sbatchman as sbm
from parsers import parse_scheduler_output

def main():
    jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED], from_active=True, from_archived=False)
    cluster_name = sbm.get_cluster_name()

    res = [parse_scheduler_output(j.get_stdout()) for j in jobs]
    for r in res:
        pprint(r)
        print()
        print('='*100)
        print()
    # out_file = OUT_DIR / f"hicrest-axccl_{cluster_name}_data.parquet"
    # import_export.describe_pairs_content(meta_df_pairs, verbose=True)
    # import_export.write_multiple_to_parquet(meta_df_pairs, out_file)


if __name__ == "__main__":
    main()

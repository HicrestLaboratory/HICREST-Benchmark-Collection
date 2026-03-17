import os
import csv
import matplotlib.pyplot as plt

# -----------------------
# Configuration
# -----------------------
input_folder = "/home/bgandham/RISC_TACO/locks/pioneer-4"  # folder containing the lock files
output_folder = "/home/bgandham/RISC_TACO/locks/pioneer-4/processed_results"

# Exact filenames mapping to locks
lock_files = {
    "FissileTicket": "FissileTicket",
    "HemLock": "HemLock",
    "MCS": "MCS",
    "PthreadLock": "PthreadLock",
    "Reciprocating": "Reciprocating",
    "SpinLock": "SpinLock"
}

os.makedirs(output_folder, exist_ok=True)

# -----------------------
# Read all files
# -----------------------
all_rows = []

for file_name, lock_name in lock_files.items():
    file_path = os.path.join(input_folder, file_name)
    if not os.path.isfile(file_path):
        print(f"WARNING: File {file_path} does not exist, skipping.")
        continue
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                row = {
                    "threads": int(parts[0]),
                    "duration": int(parts[1]),
                    "total_entries": int(parts[2]),
                    "avg_entries": float(parts[3]),
                    "std_dev": float(parts[4]),
                    "relative_std": float(parts[5].replace('%','')),
                    "lock": lock_name
                }
                all_rows.append(row)
            except ValueError:
                print(f"Skipping invalid line in {file_path}: {line}")

# -----------------------
# Write combined CSV
# -----------------------
csv_file = os.path.join(output_folder, "combined_results.csv")
csv_cols = ["threads", "lock", "duration", "total_entries", "avg_entries", "std_dev", "relative_std"]

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_cols)
    writer.writeheader()
    for r in all_rows:
        writer.writerow({k: r.get(k, 0) for k in csv_cols})

print(f"CSV generated: {csv_file}")

# -----------------------
# Plot graphs
# -----------------------
metrics = ["total_entries", "avg_entries", "std_dev", "relative_std"]
data_by_lock = {lock: [] for lock in lock_files.values()}

for row in all_rows:
    data_by_lock[row['lock']].append(row)

for metric in metrics:
    plt.figure(figsize=(8,5))
    for lock in lock_files.values():
        rows = sorted(data_by_lock[lock], key=lambda x: x['threads'])
        threads = [r['threads'] for r in rows]
        values = [r[metric] for r in rows]
        plt.plot(threads, values, marker='o', label=lock)

    plt.xlabel("Threads")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Threads")
    plt.xscale('log', base=2)
    plt.xticks([1,2,4,8,16,32,64], [1,2,4,8,16,32,64])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    graph_file = os.path.join(output_folder, f"{metric}_vs_threads.png")
    plt.savefig(graph_file)
    plt.close()
    print(f"Graph saved: {graph_file}")

print(f"All processed CSV and graphs saved in folder: '{output_folder}'")
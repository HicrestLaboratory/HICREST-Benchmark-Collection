import random
import argparse
import pandas as pd
import sys

# --- CONFIGURATION ---
CONFIG = {
    "small": {
        "tier_weight": 0.75,
        "sizes": [2],
        "sub_weights": {}
    },
    "medium": {
        "tier_weight": 0.20,
        "sizes": [4, 8, 16],
        "sub_weights": {8: 0.50} # 50% are 8, remainder (50%) split between 4 and 16
    },
    "large": {
        "tier_weight": 0.05,
        "sizes": [32, 64],
        "sub_weights": {64: 0.20} # 20% are 64, 80% are 32
    }
}

GPUS_PER_NODE = 4

def get_weighted_size(tier_name, tier_cfg):
    sizes = tier_cfg["sizes"]
    sub_weights_map = tier_cfg.get("sub_weights", {})
    
    total_specified = sum(sub_weights_map.values())
    if total_specified > 1.0 + 1e-9:
        raise ValueError(f"Sub-weights for '{tier_name}' exceed 1.0 (Total: {total_specified})")

    remaining_sizes = [s for s in sizes if s not in sub_weights_map]
    
    final_weights = []
    for s in sizes:
        if s in sub_weights_map:
            final_weights.append(sub_weights_map[s])
        else:
            # Uniformly split the remaining probability
            rem_prob = (1.0 - total_specified) / len(remaining_sizes) if remaining_sizes else 0
            final_weights.append(rem_prob)
            
    if abs(sum(final_weights) - 1.0) > 1e-9:
        raise ValueError(f"Weights for '{tier_name}' do not sum to 1.0 and no sizes available to fill.")
        
    return random.choices(sizes, weights=final_weights, k=1)[0]

def generate_cluster_data(target_nodes):
    tier_names = list(CONFIG.keys())
    tier_weights = [c["tier_weight"] for c in CONFIG.values()]
    
    if abs(sum(tier_weights) - 1.0) > 1e-9:
        raise ValueError(f"Global tier weights must sum to 1.0 (Current: {sum(tier_weights)})")

    jobs = []
    current_nodes = 0

    while current_nodes < target_nodes:
        tier = random.choices(tier_names, weights=tier_weights, k=1)[0]
        nodes = get_weighted_size(tier, CONFIG[tier])
        
        jobs.append({
            "tier": tier,
            "nodes": nodes,
            "gpus": nodes * GPUS_PER_NODE
        })
        current_nodes += nodes

    return pd.DataFrame(jobs)

def main():
    parser = argparse.ArgumentParser(description="Generate distributed jobs using pandas.")
    parser.add_argument("total_nodes", type=int, help="Target node capacity to cover")
    args = parser.parse_args()

    try:
        df = generate_cluster_data(args.total_nodes)
        
        print("\n### GENERATED JOB LIST ###")
        print(df.sort_values('gpus').to_string(index=False))
        
        print("\n" + "="*40)
        print("### SUMMARY STATISTICS ###")
        print("="*40)
        
        # Calculate summary metrics
        total_nodes = df['nodes'].sum()
        total_gpus = df['gpus'].sum()
        
        # Tier breakdown
        summary = df.groupby('tier').agg(
            job_count=('nodes', 'count'),
            total_nodes=('nodes', 'sum'),
            avg_nodes=('nodes', 'mean')
        )
        
        print(summary)
        print("-" * 40)
        print(f"Target Capacity:  {args.total_nodes} nodes")
        print(f"Actual Capacity:  {total_nodes} nodes ({(total_nodes/args.total_nodes)*100:.1f}%)")
        print(f"Total GPUs:       {total_gpus}")
        print(f"Total Jobs:       {len(df)}")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
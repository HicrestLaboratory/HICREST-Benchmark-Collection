# ============================================================================
#  Display constants
# ============================================================================

# Canonical ordering used to lay out bars within each placement group.
STRATEGY_ORDER = ["DP", "FSDP", "DP+PP", "DP+PP+TP", "DP+PP+Expert"]

# Short display names for strategies (used in tick labels and legend).
STRATEGY_DISPLAY = {
    "DP":           "D",
    "FSDP":         "FSDP",
    "DP+PP":        "D+P",
    "DP+PP+TP":     "D+P+T",
    "DP+PP+Expert": "D+P+E",
}

# Model-name display mapping (from raw model_name strings to short labels).
STRATEGY_MODEL = {
    "vit-h":        "ViT-H",
    "llama3-8b":    "LaM-8",
    "llama3-70b":   "LaM-70",
    "minerva-7b":   "Minv",
    "mixtral-8x7b": "Mxt",
}

# Canonical placement ordering (primary sort key; bars are grouped by this).
PLACEMENT_ORDER = [
    "intra-l1",
    "intra-group",
    "intra-group-same-l1-2",
    "intra-group-same-l1-4",
    "inter-group",
    "inter-group-same-l1-2",
    "inter-group-same-l1-4",
]

# One distinct colour per parallelism strategy.
STRATEGY_COLORS = {
    "DP":           "#1f77b4",   # blue
    "FSDP":         "#ff7f0e",   # orange
    "DP+PP":        "#2ca02c",   # green
    "DP+PP+TP":     "#d62728",   # red
    "DP+PP+Expert": "#9467bd",   # purple
}

# Human-readable placement names used as shared group labels on the x-axis.
PLACEMENT_DISPLAY = {
    "intra-l1":              "Intra L1",
    "intra-group":           "Intra Group",
    "inter-group":           "Inter Group",
    "intra-group-same-l1-2": "Intra Group\n2 Nodes/Switch",
    "inter-group-same-l1-2": "Inter Group\n2 Nodes/Switch",
    "intra-group-same-l1-4": "Intra Group\n4 Nodes/Switch",
    "inter-group-same-l1-4": "Inter Group\n4 Nodes/Switch",
}

GPUS_PER_NODE_MAP = {
    'dgxA100':      8,
    'jupiter':      4,
    'leonardo':     4,
    'nvl72':        4,
    'alps':         4,
    'lumi':         8,
    'cresco':       8,
}

SYSTEM_NAMES_MAP = {
    'dgxA100':      'DGX A100',
    'jupiter':      'Jupiter',
    'leonardo':     'Leonardo',
    'nvl72':        'NVL 72',
    'alps':         'Alps (Daint)',
    'lumi':         'LUMI',
    'cresco':       'Cresco',
}
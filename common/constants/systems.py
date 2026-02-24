# Name maps
BOARD_NAMES_MAP = {
  'brah': 'AMD EPYC 7742',
  'baldo': 'AMD EPYC 7742',
  'pioneer': 'Milk-V Pioneer',
  'bananaf3': 'Banana Pi F3',
  'arriesgado': 'HiFive Unmatched',
}
BOARD_SHORT_NAMES_MAP = {
  'brah': 'AMD',
  'baldo': 'AMD',
  'pioneer': 'Pioneer',
  'bananaf3': 'BananaPi',
  'arriesgado': 'HiFive',
}

# Hard-coded data
CACHE_SIZES = {
  'pioneer':    [(64 * 1024,        'L1d'), (1 * 1024 * 1024,  'L2'),  (64 * 1024 * 1024,  'L3')],
  'bananaf3':   [(32 * 1024,        'L1d'), (500 * 1024     ,  'L2'),  (2* 500 * 1024,     'L2 + TCM')],
  'arriesgado': [(64 * 1024,        'L1d'), (2 * 1024 * 1024,  'L2'),  (0,                 '')        ],
  'brah':       [(4 * 1024 * 1024,  'L1d'), (64 * 1024 * 1024, 'L2'),  (0,                 '')        ],# (512 * 1024 * 1024, 'L3')],
  'baldo':      [(4 * 1024 * 1024,  'L1d'), (64 * 1024 * 1024, 'L2'),  (0,                 '')        ],# (512 * 1024 * 1024, 'L3')],
}

CLUSTER_NAMES_MAP = {
  'nanjing': 'NJ',
  'nanjing-inter': 'NJ',
  'nanjing-intra': 'NJ',
  'haicgu': 'HAICGU',
  'leonardo': 'Leonardo',
  'local': 'local',
}

PARTITION_NAMES_MAP = {
  'ib': 'ib',
  'eth': 'eth',
  'nanjing-inter': 'inter',
  'nanjing-intra': 'intra',
  'boost_usr_prod': 'booster',
  # Debug
  'test': 'test',
  'other': 'other',
}

DEFAULT_PARTITION_NAMES_MAP = {
  CLUSTER_NAMES_MAP['leonardo']: 'booster'
}

# ==========================================================
# System Interconnect Specs (Gb/s per direction)
# ==========================================================

SYSTEM_INTERCONNECT_SPECS = {
    'leonardo': {
        'bw_gpu_gpu': 200,  # 4 x NVLink 3.0
        'bw_gpu_nic': 256,  # 1 x PCIe 4.0 x16
        'bw_cpu_nic': 256,  # 1 x PCIe 4.0 x16
        'bw_nic_l1':  100,  # Infiniband HDR
        'bw_l1_l2':   100,  # Dragonfly+
        'bw_l2_l2':   200,  # Dragonfly+
    },
    'alps': {
        'bw_gpu_gpu': 200,  # 4 x NVLink 4.0
        'bw_cpu_gpu': 3600, # NVLink C2C
        'bw_gpu_nic': 512,  # ?
        'bw_cpu_nic': 512,  # 1 x PCIe 4.0 x16
        'bw_nic_l1':  200,  # Slingshot 11
        'bw_l1_l1':   200,  # ? Slingshot 11
    }
}


# ==========================================================
# Path mapping and bottleneck computation (for bandwidth)
# ==========================================================

# TODO double-check
def get_path_links(system, comm_type, topology):
    if system in ['leonardo']: # Dragonfly+
        if comm_type == 'G2G':
            if topology == 'same-l1':
                return ['bw_gpu_gpu', 'bw_nic_l1']
            elif topology == 'same-group':
                return ['bw_gpu_nic', 'bw_nic_l1', 'bw_l1_l2']
            elif topology == 'inter-group':
                return ['bw_gpu_nic', 'bw_nic_l1', 'bw_l1_l2', 'bw_l2_l2']

        elif comm_type == 'C2C':
            if topology == 'same-l1':
                return ['bw_cpu_nic']
            elif topology == 'same-group':
                return ['bw_cpu_nic', 'bw_nic_l1', 'bw_l1_l2']
            elif topology == 'inter-group':
                return ['bw_cpu_nic', 'bw_nic_l1', 'bw_l1_l2', 'bw_l2_l2']
    
    if system in ['alps']: # Dragonfly+
        if comm_type == 'G2G':
            if topology == 'same-l1':
                return ['bw_gpu_gpu', 'bw_nic_l1']
            elif topology == 'same-group':
                return ['bw_gpu_nic', 'bw_nic_l1', 'bw_l1_l1']
            elif topology == 'inter-group':
                return ['bw_gpu_nic', 'bw_nic_l1', 'bw_l1_l1']

        elif comm_type == 'C2C':
            if topology == 'same-l1':
                return ['bw_cpu_nic']
            elif topology == 'same-group':
                return ['bw_cpu_nic', 'bw_nic_l1', 'bw_l1_l1']
            elif topology == 'inter-group':
                return ['bw_cpu_nic', 'bw_nic_l1', 'bw_l1_l1']

    return []


def compute_theoretical_bandwidth(system, comm_type, topology):
    if system not in SYSTEM_INTERCONNECT_SPECS:
        return None

    specs = SYSTEM_INTERCONNECT_SPECS[system]
    links = get_path_links(system, comm_type, topology)

    values = [specs[l] for l in links if l in specs]
    if not values:
        return None

    return min(values)

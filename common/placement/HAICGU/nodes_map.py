from dataclasses import dataclass

# IB: 09 - 18
# ETH: 19 - 28

@dataclass
class HAICGUNodesMap:
    def get_id(self, node: str) -> int:
        return int(node.split('.')[0][2:])
        
    def get_partition(self, id: int) -> str | None:
        if id >= 9 and id <= 18:
            return 'ib'
        if id >= 19 and id <= 28:
            return 'eth' 
        
    def get_node_distance(self, node1: str, node2: str) -> int:
        id1 = self.get_id(node1)
        id2 = self.get_id(node2)
        part1 = self.get_partition(id1)
        part2 = self.get_partition(id2)
        if part1 != part2:
            return 99999
        if id1 == id2:
            return 0
        return 1
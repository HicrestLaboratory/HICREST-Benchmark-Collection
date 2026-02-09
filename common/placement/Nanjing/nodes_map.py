from dataclasses import dataclass

@dataclass
class NanjingNodesMap:
    def get_id(self, node: str) -> int:
        return int(node[len('host'):])
    
    def get_switch(self, id: int) -> int:
        return 0 if id <= 234 else 1
        
    def get_node_distance(self, node1: str, node2: str) -> int:
        id1 = self.get_id(node1)
        id2 = self.get_id(node2)
        sw1 = self.get_switch(id1)
        sw2 = self.get_switch(id2)
        return 0 if sw1 == sw2 else 1
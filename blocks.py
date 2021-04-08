import math
from functools import reduce 

class Block:
    """Blocks are defined by an ordered list of 2D points.
    The first point should be (0,0) and the second should be (X,0) where X 
    the length of the edge connecting the first and second points.
    A square block would be represented by 4 points: [(0,0), (X,0), (X,X), (0,X)]
    Each pair of consecutive points defines an edge in the block.
    The final edge is formed from the last point and the first (0,0)
    """
    def __init__(self, points):
        self.vertices = points
    def get_num_edges(self):
        return len(self.vertices)

    def get_edge_points(self, index):
        if index < 0:
            raise Exception("index cannot be negative")
        index = index 
        return (self.vertices[index % self.get_num_edges()], self.vertices[(index+1)% self.get_num_edges()])

    def move_block_to_edge(self):
        pass
    
    def _center_of_gravity(self):
        ln = len(self.vertices)
        sumx = sum([x for (x,_) in self.vertices])
        sumy = sum([y for (_,y) in self.vertices])
        return (sumx/ln, sumy/ln)
    
    def _center_of_gravity2(self):
        (sumx, sumy) = reduce(lambda a,b : (a[0] + b[0], a[1] + b[1]), self.vertices, (0,0))
        ln = len(self.vertices)
        return (sumx/ln, sumy/ln)

    def _gen_id(self):
        """
        ID will be the pair.  First part is the number of edge, second part
        is the sum of  distances from each point to the block's center of gravity
        this value will be the same for a block regardless of its orientation or position
        """
        cog = self._center_of_gravity()
        return (self.get_num_edges(), sum([Block._distance(cog, pt) for pt in self.vertices]))
        
    @staticmethod
    def _distance(p,q):
        return math.sqrt((p[0] - q[0])^2 + (p[1] - q[1])^2)

    @staticmethod
    def move_and_attach(from_block, from_edge, to_block, to_edge):
        """will move the from_block over to the to_block by attaching the from_edge
        of the from_block to the to_edge of the to_block

        Args:
            from_block ([type]): [description]
            from_edge ([type]): [description]
            to_block ([type]): [description]
            to_edge ([type]): [description]

        Returns:
            Block: [description]
        """
        pass
class Blockus :
    def __init__(self, blocks = None):
        self.blocks = set(blocks) if blocks else set()

    def __len__(self):
        return len(self.blocks)
    
    def _originize(self):
        min_x = min([b.x for b in self.blocks])
        min_y = min([b.y for b in self.blocks])
        self.blocks = {Block(b.x - min_x, b.y - min_y) for b in self.blocks}

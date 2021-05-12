import math
from functools import reduce 
import numpy as np
import sys

TOLERANCE = 0.0001
viewBoxWidth=50
viewBoxHeighh=50

class Point:
    def __init__(self, xy):
        self.x = round(xy[0],4)
        self.y = round(xy[1],4)
    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    def __hash__(self):
        return hash((self.x, self.y))
    def __str__(self):
        return f'({self.x}, {self.y})'

class Distance:
    def __init__(self, d):
        self.distance = round(d, 4)
    def __eq__(self, other):
        if not isinstance(other, Distance):
            return NotImplemented
        return self.distance == other.distance
    def __hash__(self):
        return hash(self.distance)
    def __str__(self):
        return f'{self.distance}'
        
class Edge:
    def __init__(self, start_pt, end_pt):
        self.start_pt = start_pt
        self.end_pt = end_pt
    def get_points(self):
        return (self.start_pt, self.end_pt)
    def distance(self):
        x = self.start_pt[0] - self.end_pt[0]
        y = self.start_pt[1] - self.end_pt[1]
        return math.sqrt((x**2) + (y**2))
    def __eq__(self, other): 
        if not isinstance(other, Edge):
            return NotImplemented
        return (Point(self.start_pt) == Point(other.start_pt)) and (Point(self.end_pt) == Point(other.end_pt))
    def __ne__(self, other): 
        return not self.__eq__(other)

class Block:
    """Blocks are defined by an ordered array of 2D points.
    The first point should be (0,0) and the second should be (X,0) where X 
    the length of the edge connecting the first and second points.
    A square block would be represented by 4 points: [(0,0), (X,0), (X,X), (0,X)]
    Each pair of consecutive points defines an edge in the block.
    The final edge is formed from the last point and the first (0,0)
    """
    def __init__(self, points, direction=True):
        self.vertices = points
        self.direction = direction
    
    def copy(self):
        return Block(self.vertices, self.direction)
        
    def flip(self, edge):
        """ flips the current block along the specified edge

            this is from:  https://stackoverflow.com/a/3307181

            given point (x1, y1)
            and a reflection line passing through points (x2,y2) and (x3,y3)
            (x4, y4) is the mirror of point (x1,y1) across this reflection line with this formula:
            
            (x4,y4) = (2*d - x1, 2*d*m - y1 + 2*c)

            where
            m is (y3-y2)/(x3-x2)
            c is (x3*y2-x2*y3)/(x3-x2)
            d = (x1 + (y1 - c)*m)/(1 + m**2)

        """
        def flip_point(edge):
            (a,b) = edge.get_points()
            (x2,y2) = (a[0], a[1])
            (x3,y3) = (b[0], b[1])

            m = (y3-y2)/(x3-x2)
            c = (x3*y2-x2*y3)/(x3-x2)
            m2 = (1 + m**2)

            return lambda p: [2*((p[0] + (p[1] - c)*m)/m2) - p[0], 2*((p[0] + (p[1] - c)*m)/m2)*m - p[1] + 2*c]
        
        flip_point_closure = flip_point(edge)
        flipped_pts = np.array(list(map(flip_point_closure, self.vertices)))
        return Block(flipped_pts, not self.direction)

    def flip2(self, a, b):

        """[summary]
        another formula from: https://youtu.be/SD_qew7vOtw
        given a line y = ax + b
        then a given point (x,y) will be reflected over this line to point (x',y') by applying this
        x' = ((1-a**2)/(1+a**2))x + (2*a/(a**2 + 1))(y-b)
        y' = (((a**2 + 1)/(a**2 + 1)))*(y-b) + ((2*a*x)/(a**2 + 1)) + b
        """
        def xf(x,y): 
            return ((1-a**2)/(1+a**2))* x + (2*a/(a**2 + 1))*(y-b)
        
        def yf(x,y):
            return (((a**2 - 1)/(a**2 + 1)))*(y-b) + ((2*a*x)/(a**2 + 1)) + b

        flipped_pts = np.array(list(map(lambda p: [xf(p[0],p[1]), yf(p[0],p[1])], self.vertices)))
        return Block(flipped_pts, not self.direction)

    def num_edges(self):
        return len(self.vertices)

    def get_edge(self, index):
        if index < 0:
            raise Exception("index cannot be negative")
        index = index 
        return Edge (self.vertices[index % self.num_edges()], self.vertices[(index+1)% self.num_edges()])

    def _center_of_gravity(self):
        (sumx, sumy) = reduce(lambda a,b : (a[0]+b[0], a[1]+b[1]), self.vertices, (0,0))
        ln = len(self.vertices)
        return (sumx/ln, sumy/ln)

    def __str__(self):
        points = ""
        first=True
        for pt in self.vertices:
            sep="" if first else ", "
            first=False
            points=f"{points}{sep}[{pt[0]} {pt[1]}]"

        return f'Block: {"+" if self.direction else "-"}[{points}]'

    def _gen_id(self):
        """
        ID will be the pair.  First part is the number of edge, second part
        is the sum of  distances from each point to the block's center of gravity
        this value will be the same for a block regardless of its orientation or position
        """
        def _distance(p,q):
            return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
        cog = self._center_of_gravity()
        return (self.num_edges(), sum([Block._distance(cog, pt) for pt in self.vertices]))
        
    def align_block_on_edge(self, fixed_block_edge, moveable_block, moveable_block_edge):
        """ will attach two blocks by connecting them together at the specified edges
        the current block (self) will remain in place, and the moveable_block will be 
        translated and rotated as necessary for the new blocks to affix to each other at the edge
        This method will verify that the lengths of the edges are the same.
        This method will verify that the blocks do not overlap after they are joined.
        This method will return a new block representing the newly formed and co-joined block

        Args:
            block1 ([type]): [description]
            block1_edge ([type]): [description]
            block2 ([type]): [description]
            block2_edge ([type]): [description]

        Returns:
            Block: [descrption]
        """
        fixed_block = self

        f_edge = fixed_block.get_edge(fixed_block_edge)
        m_edge = moveable_block.get_edge(moveable_block_edge)

        (f_pt_a, f_pt_b) = f_edge.get_points()
        (m_pt_a, m_pt_b) = m_edge.get_points()
        lfedge = f_edge.distance()
        lmedge = m_edge.distance()
        if Distance(lfedge) != Distance(lmedge):
            raise  Exception("edges were not the same length")

        pivot_pt = f_pt_a
        move_to_pt = f_pt_a
        rotate_to_pt = f_pt_b

        if fixed_block.direction == moveable_block.direction:
            move_from_pt = m_pt_b
            rotate_from_pt = m_pt_a
        else:
            move_from_pt = m_pt_a
            rotate_from_pt = m_pt_b

        translated_block = moveable_block.move(from_=move_from_pt, to_=move_to_pt)
        
        # move the point on the edge to be rotated
        # rotate_from_pt =  Block.move_point(rotate_from_pt, from_=move_from_pt, to_=move_to_pt)
        rotate_from_pt =  rotate_from_pt - (move_from_pt - move_to_pt)
        
        rotated_and_translated_block = translated_block.rotate(pivot_=pivot_pt, from_=rotate_from_pt, to_=rotate_to_pt)
        final_merged_block = fixed_block.merge(rotated_and_translated_block)
        #return final_merged_block
        return (translated_block, rotated_and_translated_block)
        #return 

    def merge(self, other_block):
        return other_block


    @staticmethod
    def find_shared_pts(points_a, points_b):
        a = { Point(p) for p in points_a}
        b = { Point(p) for p in points_b}
        return a.intersection(b)        

    def merge_blocks(self, other_block):
        """[summary]

        Args:
            other_block ([type]): [description]

        Returns:
            [type]: [description]

        Yields:
            [type]: [description]
        """
        shared_pts = Block.find_shared_pts(self.vertices, other_block.vertices)


    @staticmethod
    def draw_blocks(blocks, file=sys.stdout):
        print(f'<svg width="800" height="800" viewBox="0 0 {viewBoxWidth} {viewBoxHeighh}" preserveAspectRatio="xMinYMin meet" xmlns="http://www.w3.org/2000/svg">', file=file)
        cols = ['red', 'green', 'orange', 'blue', 'brown']
        i = 0
        for b in blocks:
            b.draw_svg(file=file, color=cols[i] )
            i += 1
        print('</svg>', file=file)        
    
    def draw_svg(self, title=None, color='black', file=sys.stdout, wrap=False):
        points=""
        first=True
        # scale=100
        scale=1
        label=None

        if title:
            label = f'<text x="{int(self.vertices[2][0]*scale)}" y="{int(self.vertices[2][1]*scale)}">{title}</text>'
            print(label, file=file)

        print(f'<polygon points="0 0, 0 {viewBoxHeighh}, {viewBoxWidth} {viewBoxHeighh}, {viewBoxWidth} 0" fill="none" style="stroke:black;stroke-width:0.1"/>', file=file)

        for pt in self.vertices:
            if first:
                sep=""
                first=False
            else:
                sep=", "
            inv_pt = viewBoxHeighh - pt[1]
            points=f"{points}{sep}{int(pt[0]*scale)} {int(inv_pt*scale)}"

        print(f'<polygon points="{points}" fill="none" style="stroke:{color};stroke-width:0.1"/>', file=file)

    def move(self, from_, to_):
        b = Block(self.vertices, self.direction)
        move_delta = from_ - to_
        b.vertices = b.vertices - move_delta
        return b

    def rotate(self, pivot_, from_, to_):
        b = self.copy()
        theta = Block.calc_angle(pivot_, from_, to_)
        b.rotate2(pivot_, -theta)
        return b

    def rotate2(self, rotation_pt, theta):
        # create the rotation matrix given the angle of rotation theta
        rot_matrix = np.array([[math.cos(theta), - math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        
        # translate the block such that the rotation point is at the origin
        translate_to_origin = np.array([0.0, 0.0]) - rotation_pt
        self.vertices = self.vertices + translate_to_origin
        
        # now rotate
        rotated_block = self.vertices @ rot_matrix

        # now translate the block back to the rotation point's original location
        self.vertices = rotated_block - translate_to_origin

    @staticmethod
    def calc_angle(vertex_pt, start_pt, end_pt):
        a = start_pt - vertex_pt
        b = end_pt - vertex_pt
        ab_dot = a.dot(b)
        len_a = math.sqrt(a[0]**2 + a[1]**2)
        len_b = math.sqrt(b[0]**2 + b[1]**2)
        return math.acos(ab_dot/(len_a * len_b))
##########################################################################
# only functions below

def flip(block, edge):
    """ flips the current block along the specified edge

        this is from:  https://stackoverflow.com/a/3307181

        given point (x1, y1)
        and a reflection line passing through points (x2,y2) and (x3,y3)
        (x4, y4) is the mirror of point (x1,y1) across this reflection line with this formula:
        
        (x4,y4) = (2*d - x1, 2*d*m - y1 + 2*c)

        where
        m is (y3-y2)/(x3-x2)
        c is (x3*y2-x2*y3)/(x3-x2)
        d = (x1 + (y1 - c)*m)/(1 + m**2)

    """
    def flip_point(edge):
        (a,b) = edge.get_points()
        (x2,y2) = (a[0], a[1])
        (x3,y3) = (b[0], b[1])

        m = (y3-y2)/(x3-x2)
        c = (x3*y2-x2*y3)/(x3-x2)
        m2 = (1 + m**2)

        return lambda p: [2*((p[0] + (p[1] - c)*m)/m2) - p[0], 2*((p[0] + (p[1] - c)*m)/m2)*m - p[1] + 2*c]

    flip_point_closure = flip_point(edge)
    flipped_pts = np.array(list(map(flip_point_closure, block.vertices)))
    return Block(flipped_pts, not block.direction)

def flip2(block, a, b):

    """[summary]
    another formula from: https://youtu.be/SD_qew7vOtw
    given a line y = ax + b
    then a given point (x,y) will be reflected over this line to point (x',y') by applying this
    x' = ((1-a**2)/(1+a**2))x + (2*a/(a**2 + 1))(y-b)
    y' = (((a**2 + 1)/(a**2 + 1)))*(y-b) + ((2*a*x)/(a**2 + 1)) + b
    """
    def xf(x,y): 
        return ((1-a**2)/(1+a**2))* x + (2*a/(a**2 + 1))*(y-b)

    def yf(x,y):
        return (((a**2 - 1)/(a**2 + 1)))*(y-b) + ((2*a*x)/(a**2 + 1)) + b

    flipped_pts = np.array(list(map(lambda p: [xf(p[0],p[1]), yf(p[0],p[1])], block.vertices)))
    return Block(flipped_pts, not block.direction)

if __name__ == '__main__':

    pass

    # @staticmethod
    # def get_iter_between_items(arr, item1, item2):
    #     have_reached_item1 = False
    #     INF_LOOP_GUARD = 100
    #     for item in Block.circular_iter(arr):
    #         if INF_LOOP_GUARD < 0:
    #             break
    #         else:
    #             INF_LOOP_GUARD -= 1
    #         if not have_reached_item1:
    #             if item == item1:
    #                 have_reached_item1 = True
    #         else:
    #             yield item
    #             if item == item2:
    #                 break
    #
    # def circular_iter(arr):
    #     alen = len(arr)
    #     indx = alen
    #     while True:
    #         if indx < alen:
    #             yield arr[indx]
    #             indx = indx + 1
    #         else:
    #             indx = 0


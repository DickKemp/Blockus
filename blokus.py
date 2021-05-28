from __future__ import annotations
import math
from typing import List, Set, Tuple 
import numpy as np

from point import Point, calc_angle, rearrange_origin, is_path_clockwise, is_eq_float, get_item, find_shared_pts

from edge import Edge

class Blok:
    """a Blok is an ordered array of 2D points that describes the polygon outline
    of the block.  Constructs a block by passing in the points of the polygon.
    the initializer determines the direction in which the points circumscribe the polygon, and 
    then stores the clockwise version of the points 

    E.g., to define a square block you can pass in these points {(0,0), (X,0), (X,X), (0,X)]
    Each pair of consecutive points defines an edge in the block.
    The final edge is formed from the last point and the first
    after checking the direction of the points, it would determine that they are going 
    counterclockwise, so the init method would reverse the list and store the points as
    [(0,X), (X,X), (X,0), (0,0)]
    """

    # Blok() is a constructor for a new Blok
    # Blok(points: List[Point]) -> Blok

    def __init__(self: Blok, points: List[Point]) -> None:
        is_clockwise, path_origin = Blok.find_clockwise_origin(points)
        points = points if is_clockwise else points[::-1]
        self.points = rearrange_origin(points, path_origin)
        self.num_edges = len(points)    
    
    def get_nparray(self : Blok) -> np.ndarray:
        return np.array([[p.x,p.y] for p in self.points])

    def __eq__(self, other):
        if not isinstance(other, Blok):
            return NotImplemented
        if len(self.points) != len(other.points):
            return False
        for i in range(len(self.points)):
            if self.points[i] != other.points[i]:
                return False
        return True

    def __str__(self: Blok) -> str:
        points = ""
        first=True
        for pt in self.points:
            sep="" if first else ", "
            first=False
            points=f"{points}{sep}{pt}"
        return f'Blok:[{points}]'

    @staticmethod
    def create_from_nparray(nparray: np.ndarray) -> Blok:
        return Blok([Point(p[0],p[1]) for p in nparray])

    # num_edges() returns the number of edges in a block
    @staticmethod
    def num_edges(b: Blok) -> int:
        return b.num_edges

    # get_edge() returns the 0-indexed edge ei from block b
    @staticmethod
    def get_edge(b: Blok, ei: int) -> Edge:
        p0 = b.points[ei % b.num_edges]
        p1 = b.points[(ei+1) % b.num_edges]
        return Edge(Point(p0.x, p0.y), Point(p1.x, p1.y))

    # normalize() will translate a block so that it is positioned near the origin
    @staticmethod
    def normalize(b: Blok) -> Blok:
        min_x = min(p.x for p in b.points)
        min_y = min(p.y for p in b.points)
        return Blok.translate(b, -min_x, -min_y)

    # translate() will translate a block to a new position
    @staticmethod
    def translate(b: Blok, x: float, y: float) -> Blok:
        moved_points = list(map(lambda p: Point.translate(p, x, y), b.points))
        return Blok(moved_points)
    
    # rotate() will rotate the block theta radians around the point p
    @staticmethod
    def rotate(b: Blok, rp: Point, theta: float) -> Blok:
        # translate the block such that the rotation point is at the origin
        b2 = Blok.translate(b, -rp.x, -rp.y)

        # create the rotation matrix given the angle of rotation theta
        rot_matrix = np.array([[math.cos(theta), - math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        
        rotated_block_points_array = b2.get_nparray() @ rot_matrix
        
        b3 = Blok.create_from_nparray(rotated_block_points_array)

        # translate the block back to the rotation point's original location
        return Blok.translate(b3, rp.x, rp.y)


    @staticmethod
    def rotate_to_point(b, pivot_, from_, to_):
        theta = calc_angle(pivot_, from_, to_)
        return Blok.rotate(b, pivot_, -theta)

    # flip()
    @staticmethod
    def flip(b: Blok, edge_: Edge) -> Blok:

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
        def flip_point_func(edge: Edge):
            a = edge.start_pt
            b = edge.end_pt
            (x2,y2) = (a.x, a.y)
            (x3,y3) = (b.x, b.y)

            m = (y3-y2)/(x3-x2)
            c = (x3*y2-x2*y3)/(x3-x2)
            m2 = (1 + m**2)

            return lambda p: Point(2*((p.x + (p.y - c)*m)/m2) - p.x, 2*((p.x + (p.y - c)*m)/m2)*m - p.y + 2*c)

        flip_point_closure = flip_point_func(edge_)
        return Blok(list(map(flip_point_closure, b.points)))

    # copy()
    @staticmethod
    def copy(b: Blok) -> Blok:
        return Blok(b.points)

    # align_blocks_on_edge() will rotate and translate 2 blocks
    # so that the e1 edge of block b1 overlaps exactly with the e2 edge of block b2
    # definition: two edges overlap if the edges are equal
    # will raise an exception if len(e1) != len(e2)
    @staticmethod
    def align_blocks_on_edge(b1: Blok, e1: int, b2: Blok, e2: int) -> Tuple[Blok, Blok]:    

        b1_edge = Blok.get_edge(b1, e1)
        b2_edge = Blok.get_edge(b2, e2)

        if not Edge.has_same_length(b1_edge, b2_edge):
            raise  Exception("edges were not the same length")

        from_pt = b2_edge.end_pt
        to_pt = b1_edge.start_pt

        delta_x = to_pt.x - from_pt.x
        delta_y = to_pt.y - from_pt.y

        translated_block = Blok.translate(b2, delta_x, delta_y)

        # we need to check that direction of the path from rotate_to_pt -> pivot -> rotate_from_pt
        # if clockwise, then we rotate the computed angle in the positive direction, i.e. clockwise
        # else, we rotate the compute angle in teh negative direction

        rotate_from_pt = Blok.get_edge(translated_block, e2).start_pt
        rotation_pivot_pt = b1_edge.start_pt
        rotate_to_pt = b1_edge.end_pt

        if is_path_clockwise(rotate_to_pt, rotation_pivot_pt, rotate_from_pt):
            rotate_direction = 1
        else:
            rotate_direction = -1

        theta = calc_angle(rotation_pivot_pt, rotate_from_pt, rotate_to_pt)
        
        rotated_and_translated_block = Blok.rotate(translated_block, rotation_pivot_pt, rotate_direction * theta)

        return (b1, rotated_and_translated_block)

    # is_same_block() returns true if the blocks b1 and b2 have exactly the same size and shape
    @staticmethod
    def is_same_block(b1: Blok, b2: Blok) -> bool:
        pass

    # merge() will return the block formed after merging blocks b1 and b2
    # b1 and b2 are retained in the resulting block as component blocks
    @staticmethod
    def merge(b1: Blok, b2: Blok) -> Blok:
        shared_pts = find_shared_pts(b1.points, b2.points)
        raise  Exception("not implemented")

    @staticmethod    
    def _find_convex_hull_vertex(points: List[Point]) -> Tuple[Point, Point, Point]:
        # points is an array of Point that circumscribes a polygon in either the clockwise of counterclockwise
        # direction; find_convex_hull_vertex() will find and return 3 points of the convex hull of that polygon by
        # simply finding the indices of points with the smallest Y component, and if there are more than one, 
        # picking the one with the smallest X component

        points_e = list(enumerate(points))
        min_y_pt_e = min(points_e, key=lambda p: p[1].y)
        min_py = [(i,p) for i,p in points_e if is_eq_float(p.y, min_y_pt_e[1].y)]
        vertex = min(min_py, key=lambda pi: pi[1].x)
        vertex_index = vertex[0]
        return ( get_item(points, vertex_index-1),
                 get_item(points, vertex_index),
                 get_item(points, vertex_index+1))

    # returns a tuple, the first is a bool of whether the path is clockwise or not, 
    # and the second is the lowest most left point which which be a good standard starting
    # point for the origin of the block
    @staticmethod
    def find_clockwise_origin(pts: List[Point]) -> Tuple[bool, Point]:

        a,origin,c = Blok._find_convex_hull_vertex(pts)
        return is_path_clockwise(a,origin,c), origin

if __name__ == '__main__':

    pass

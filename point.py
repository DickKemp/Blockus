from __future__ import annotations
from typing import List, Set, Tuple 
import math
import numpy as np

class Point:
    TOLERANCE = 0.0001
    def __init__(self: Point, x: float, y: float) -> None:
        self.x = x
        self.y = y
    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        #return abs(self.x - other.x) < Point.TOLERANCE and abs(self.y - other.y) < Point.TOLERANCE
        # return self.x == other.x and self.y == other.y
        return round(self.x,4) == round(other.x,4) and round(self.y,4) == round(other.y,4)
        
    def get_nparray(self):
        return np.array([self.x, self.y])

    def __hash__(self):
        return hash((round(self.x, 4), round(self.y,4)))

    def __str__(self):
        return f'({round(self.x, 4)}, {round(self.y, 4)})'
    def __repr__(self):
        return f'({round(self.x, 4)}, {round(self.y, 4)})'

    @staticmethod
    def translate(pt: Point, x: float, y: float) -> Point:
        return Point(pt.x + x, pt.y + y)


def calc_angle(vertex_pt: Point, start_pt: Point, end_pt: Point):
    # formula from: https://www.wikihow.com/Find-the-Angle-Between-Two-Vectors
    a = start_pt.get_nparray() - vertex_pt.get_nparray()
    b = end_pt.get_nparray() - vertex_pt.get_nparray()
    ab_dot = a.dot(b)
    len_a = math.sqrt(a[0]**2 + a[1]**2)
    len_b = math.sqrt(b[0]**2 + b[1]**2)
    return math.acos(ab_dot/(len_a * len_b))

def center_of_gravity(points: List[Point]) -> Point:
    pass

def distance_between_pts(p1: Point, p2: Point) -> float:
        x = p1.x - p2.x
        y = p1.y - p2.y
        return math.sqrt((x**2) + (y**2))    

def gen_id(points: List[Point]) -> Tuple[int, float]:
    """
    ID will be the pair.  First part is the number of edge, second part
    is the sum of  distances from each point to the block's center of gravity
    this value will be the same for a block regardless of its orientation or position
    """
    def _distance(p,q):
        return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
    cog = center_of_gravity(points)
    return (len(points), sum([distance_between_pts(cog, pt) for pt in points]))

def find_shared_pts(pts1: List[Point], pts2: List[Point]) -> Tuple(Set(Point), Set(Point), Set(Point)):
    a = { Point(p.x, p.y) for p in pts1 }
    b = { Point(p.x, p.y) for p in pts2 }
    both_ab = a.intersection(b)
    just_a = a.difference(both_ab)
    just_b = b.difference(both_ab)
    return (just_a, both_ab, just_b) 

def rearrange_origin(points: List[Point], pt: Point) -> List[Point]:
    indx = points.index(pt)
    rearranged = []
    for i in range(len(points)):
        rearranged.append(get_item(points, indx + i))
    return rearranged

def is_path_clockwise(a: Point,b: Point, c:Point) -> bool:
    # see orientation of a simple polygon for formula 
    # here: https://en.wikipedia.org/wiki/Curve_orientation
    det = ((b.x*c.y) + (a.x*b.y) + (a.y*c.x)) - ((a.y*b.x) + (b.y*c.x) + (a.x*c.y))
    return det < 0

def get_item(seq, n):
    len_seq = len(seq)
    return seq[n % len_seq]

def is_eq_float(f1: float, f2: float) -> bool:
    return abs(f1 - f2) < Point.TOLERANCE

def diff(pt1:Point, pt2:Point)->Point:
    return Point(pt1.x - pt2.x, pt1.y - pt2.y)

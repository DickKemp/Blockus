from __future__ import annotations
import numpy as np
from point import Point, distance_between_pts
import math

class Edge:
    def __init__(self: Edge, start_pt: Point, end_pt: Point) -> None:
        self.start_pt = start_pt
        self.end_pt = end_pt

    def __eq__(self, other): 
        if not isinstance(other, Edge):
            return NotImplemented
        return (self.start_pt == other.start_pt) and (self.end_pt == other.end_pt)

    def __ne__(self, other): 
        return not self.__eq__(other)
    
    @staticmethod
    def length(e: Edge) -> float:
        return distance_between_pts(e.start_pt, e.end_pt)

    @staticmethod
    def has_same_length(e1: Edge, e2: Edge) -> bool:
        diff = Edge.length(e1) - Edge.length(e2)
        return abs(diff) < Point.TOLERANCE
    


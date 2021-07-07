from __future__ import annotations
import math
from typing import List, Set, Tuple 
import numpy as np
from functools import reduce

from point import Point, calc_angle, diff, rearrange_origin, is_path_clockwise, is_eq_float, get_item, find_shared_pts, distance_between_pts, center_of_gravity, reorder

from canvas import Shape, Box, PolygonShape, Canvas, Style

from edge import Edge

class Blok:
    """a Blok is an ordered array of 2D points that describes the polygon outline
    of the block.  Constructs a block by passing in the points of the polygon.
    the initializer determines the direction in which the points circumscribe the polygon, and 
    then stores the clockwise version of the points 

    E.g., to define a square block you can pass in these points [ Point(0,0), Point(1,0), Point(1,1), Point(0,1)]
    Each pair of consecutive points defines an edge in the block.
    The final edge is formed from the last point and the first
    after checking the direction of the points, it would determine that they are going 
    counterclockwise, so the init method would reverse the list and store the points as
    [Point(0,0), Point(0,1), Point(1,1), Point(1,0)]
    Note that the initializer also finds the point with the smallest y-value (and if there more multiple
    points with the smallest y-value, then chooses the point with the smallest x-value) and rearranges
    the list so that that point is first in the list
    """

    def __init__(self: Blok, points: List[Point], component_blocks: List[Blok]=None) -> None:
        is_clockwise, path_origin = Blok.find_clockwise_origin(points)
        points = points if is_clockwise else points[::-1]
        self.points = rearrange_origin(points, path_origin)
        self.num_edges = len(points)    
        self.component_blocks = component_blocks

    def get_nparray(self : Blok) -> np.ndarray:
        return np.array([[p.x,p.y] for p in self.points])

        """redefines __eq__ so that two blocks are determined to be equal if they are the 
        same block regardless of orientation or position.
        """
    def __eq__(self, other):
        if not isinstance(other, Blok):
            return NotImplemented
        return Blok.is_same_block(self, other)

    def __hash__(self):
        return hash(self.gen_id())

    # is_same_block() returns true if the blocks b1 and b2 have exactly the same size and shape
    @staticmethod
    def is_same_block(b1: Blok, b2: Blok) -> bool:
        if Blok.num_edges(b1) == Blok.num_edges(b2):
            b1_diffs = b1.calc_diffs_from_cog()
            b2_diffs = b2.calc_diffs_from_cog()
            b1sum = round(sum(b1_diffs),8)
            b2sum = round(sum(b2_diffs),8)
            if b1sum == b2sum:
                return check_if_circular_lists_are_equal(b1_diffs, b2_diffs)
        return False

    def gen_id(self) -> Tuple[int, float]:
        """
        ID will be the pair.  First part is the number of edges, second part
        is the sum of  distances from each point to the block's center of gravity
        this value will be the same for a block regardless of its orientation or position
        """
        cog = center_of_gravity(self.points)
        diffs_from_cog = [distance_between_pts(cog, pt) for pt in self.points]
        diffsum = round(sum(diffs_from_cog), 8)
        return (len(self.points), diffsum)

    def calc_diffs_from_cog(self) -> List[float]:
        cog = center_of_gravity(self.points)
        diffs_from_cog = [round(distance_between_pts(cog, pt),8) for pt in self.points]
        return diffs_from_cog

    @staticmethod
    def find_flip_edge(b1: Blok) -> Edge:
        for ei in range(Blok.num_edges(b1)):
            edge = Blok.get_edge(b1,ei)
            if not Edge.is_vertical(edge):
                return edge

    def __str__(self: Blok) -> str:
        points = ""
        first=True
        for pt in self.points:
            sep="" if first else ", "
            first=False
            points=f"{points}{sep}{pt}"
        return f'Blok:[{points}]'
    
    def get_bounding_box(self: Blok) -> Box:
        minx = min([p.x for p in self.points])
        miny = min([p.y for p in self.points])
        maxx = max([p.x for p in self.points])
        maxy = max([p.y for p in self.points])
        return (Box(Point(minx, miny), Point(maxx, maxy)))

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
        component_blocks = None
        if b.component_blocks:
            component_blocks = [Blok.translate(cb, x, y) for cb in b.component_blocks]
        return Blok(moved_points, component_blocks)
    
    # rotate() will rotate the block theta radians around the point p
    @staticmethod
    def rotate(b: Blok, rp: Point, theta: float) -> Blok:
        # translate the block such that the rotation point is at the origin
        b2 = Blok.translate(b, -rp.x, -rp.y)

        # create the rotation matrix given the angle of rotation theta
        rot_matrix = np.array([[math.cos(theta), - math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        
        # to determine the location of the points after rotation, matrix multiple the points
        # of the block with the rotation matrx
        rotated_block_points_array = b2.get_nparray() @ rot_matrix
        
        # create a new blok using the rotated points
        b3 = Blok.create_from_nparray(rotated_block_points_array)

        if b.component_blocks:
            component_blocks = [Blok.rotate(cb, rp, theta) for cb in b.component_blocks]
            b3 = Blok(b3.points, component_blocks)

        # finally translate the block back to the rotation point's original location
        return Blok.translate(b3, rp.x, rp.y)
    

    @staticmethod
    def blocks_overlap(b1: Blok, b2: Blok) -> bool:
        import shapely.geometry as geo
        p1 = geo.Polygon([(round(p.x,8),round(p.y,8)) for p in b1.points])
        p2 = geo.Polygon([(round(p.x,8),round(p.y,8)) for p in b2.points])
        return p1.intersects(p1) and not p1.touches(p2)
    
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
            # a little helper function that will return a 
            # function that can be applied to a set of points to flip those
            # points to the other side of an edge acting as a reflection line
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
        return Blok(b.points, b.component_blocks)

    # align_blocks_on_edge() will rotate and translate 2 blocks
    # so that the e1 edge of block b1 overlaps exactly with the e2 edge of block b2
    # definition: two edges overlap if the edges are equal
    # will raise an exception if len(e1) != len(e2)
    @staticmethod
    def align_blocks_on_edge(b1: Blok, e1: int, b2: Blok, e2: int) -> Tuple[Blok, Blok]:    

        b1_edge = Blok.get_edge(b1, e1)
        b2_edge = Blok.get_edge(b2, e2)

        if not Edge.has_same_length(b1_edge, b2_edge):
            raise UnequalEdgesException()

        from_pt = b2_edge.end_pt
        to_pt = b1_edge.start_pt

        delta_x = to_pt.x - from_pt.x
        delta_y = to_pt.y - from_pt.y

        translated_block = Blok.translate(b2, delta_x, delta_y)

        # we need to check that direction of the path from rotate_to_pt -> pivot -> rotate_from_pt
        # if clockwise, then we rotate the computed angle in the positive direction, i.e. clockwise
        # else, we rotate the compute angle in the negative direction

        rotate_from_pt = Blok.get_edge(translated_block, e2).start_pt
        rotation_pivot_pt = b1_edge.start_pt
        rotate_to_pt = b1_edge.end_pt

        if is_path_clockwise(rotate_to_pt, rotation_pivot_pt, rotate_from_pt):
            rotate_direction = 1
        else:
            rotate_direction = -1

        theta = calc_angle(rotation_pivot_pt, rotate_from_pt, rotate_to_pt)
        
        rotated_and_translated_block = Blok.rotate(translated_block, rotation_pivot_pt, rotate_direction * theta)

        if Blok.blocks_overlap(b1, rotated_and_translated_block):
            raise OverlapException()

        return (b1, rotated_and_translated_block)


    # merge() will return the block formed after merging blocks b1 and b2
    @staticmethod
    def merge(b1: Blok, b2: Blok) -> Blok:
        only_b1, shared, only_b2 = find_shared_pts(b1.points, b2.points)

        point_only_in_b1 = next(iter(only_b1))

        merged_points = Blok._merge(b1, b2, shared, point_only_in_b1)

        # to handle the case where there may be a hole formed within the merged block, we need 
        # to make sure that the resulting merged block is truly the outer perimeter, and not the 
        # perimeter of the inner "hole"
        #
        # to consider the possibility that there is a hole, first check that the merged points include
        # each and every point in only_b1.  There will be a hole if there are points in only_b1 that are
        # not in the merged perimeter.
        # if ths is the case the pick one of those points and use it as the seed to find the merged
        # perimeter.  It should return another set of merged points. 
        # the set of points we want is the larger one that encloses the holes
        #
        # TODO:  there may be multiple holes, we should iterate and check for all

        merged_points = Blok._merge(b1, b2, shared, point_only_in_b1)

        possible_hole = only_b1 - set(merged_points)
        if len(possible_hole) > 0:
            another_point_only_in_b1 = next(iter(possible_hole))
            other_merged_points = Blok._merge(b1, b2, shared, another_point_only_in_b1)
            if len(other_merged_points) > len(merged_points):
                merged_points = other_merged_points

        return Blok(merged_points, component_blocks=[b1, b2])

    @staticmethod
    def _merge(b1, b2, shared, point_only_in_b1):
        merged_points:List[Point] = []
        bp1:List[Point] = b1.points
        bp2:List[Point] = b2.points
        i1 = 0
        i2 = 0

        # the algorithm to merge is to start creating the perimeter of points that
        # define the outline of the merged block by starting with the points in bp1 and 
        # moving forward until we hit the point that is shared with bp2.  Then we
        # switch over to moving forward in bp2 until we again hit the shared point,
        # then we switch back over to bp1 until we get back to the start.  
        # the path of points that we traversed will be the resulting merged block

    
        # 1. find the index of a point in bp1 that is not in shared, and this 
        # becomes the starting point of the new merged block
        i1 = bp1.index(point_only_in_b1)
        start = bp1[i1]

        # 2. now march along the b1 points until we hit the shared points
        # each point we traverse becomes part of the newly merged block
        while get_item(bp1, i1) not in shared:
            merged_points.append(get_item(bp1, i1))
            i1 = i1+1

        # 3. now that we reached a point that is shared, switch over to traversing over
        # the b2 points.
        # first find the index of the point in bp2 equal to the shared point that we
        # just encountered
        i2 = bp2.index(get_item(bp1, i1))

        # 4. add it to the new block
        merged_points.append(get_item(bp2, i2))
        i2 = i2 + 1

        # 5. now march along the b2 points until we hit the shared points once again
        # each point we traverse becomes part of the newly merged block
        while get_item(bp2, i2) not in shared:
            merged_points.append(get_item(bp2, i2))
            i2 = i2+1

        # 6. now that we again reached a point that is shared, switch over to traversing over
        # the b1 points
        # first find the index of the point in bp1 equal to the shared point that we
        # just encountered
        i1 = bp1.index(get_item(bp2, i2))

        # 7. add it to the new block
        merged_points.append(get_item(bp1, i1))
        i1 = i1 + 1

        # 8. now finish going through bp1 until we get back to the start point
        while get_item(bp1, i1) != start:
            merged_points.append(get_item(bp1, i1))
            i1 = i1+1
        return merged_points

    @staticmethod    
    def _find_convex_hull_vertex(points: List[Point]) -> Tuple[Point, Point, Point]:
        # points is an array of Point that circumscribes a polygon in  the clockwise
        # direction; find_convex_hull_vertex() will find and return 3 points of the convex hull of that polygon by
        # finding the index of the point with the smallest Y component (and if there are more than one, 
        # picking the one with the smallest X component).  THat points is the vertex of the convex hull.
        # The function then returns a tuple of 3 points consisting of
        # 1. the point before the vertex point, 2. the vertex point, and 3. the point after the vertex
        
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

class OverlapException(Exception):
    pass

class UnequalEdgesException(Exception):
    pass

class Circular():
    def __init__(self, lst, start_index):
        self.lst = lst
        self.start_index = start_index
        self.listlen = len(lst)
        self.item_index = 0

    def __iter__(self):
        self.item_index = 0
        return self

    def __next__(self):
        if self.item_index < self.listlen:
            val = get_item(self.lst, self.start_index + self.item_index)
            self.item_index += 1
            return val
        else:
            raise StopIteration

def check_if_circular_lists_are_equal(s1, s2):
    c1 = Circular(s1,0)
    for idx in range(len(s2)):
        c2 = Circular(s2, idx)
        if equal_sequence(iter(c1), iter(c2)):
            return True
    return False

def equal_sequence(it1, it2) -> bool:
    return tuple(it1) == tuple(it2)


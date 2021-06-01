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
        return Blok.is_same_block(self, other)

    def __hash__(self):
        return hash(self.gen_id())
        # return hash(self.gen_id3())

    # is_same_block() returns true if the blocks b1 and b2 have exactly the same size and shape
    @staticmethod
    def is_same_block0(b1: Blok, b2: Blok) -> bool:
        b1_cnt, b1_id = b1.gen_id()
        b2_cnt, b2_id = b2.gen_id()
        if b1_cnt == b2_cnt and b1_id == b2_id:
            bp1r = [round(p,8) for p in b1.points]
            bp2r = [round(p,8) for p in b2.points]
            return search_if_same_circular_lists(bp1r, bp2r)
        return False

    # is_same_block() returns true if the blocks b1 and b2 have exactly the same size and shape
    @staticmethod
    def is_same_block(b1: Blok, b2: Blok) -> bool:
        if Blok.num_edges(b1) == Blok.num_edges(b2):
            b1_diffs = b1.calc_diffs_from_cog()
            b2_diffs = b2.calc_diffs_from_cog()
            b1sum = round(sum(b1_diffs),8)
            b2sum = round(sum(b2_diffs),8)
            if b1sum == b2sum:
                return search_if_same_circular_lists(b1_diffs, b2_diffs)
        return False

    def gen_id3(self) -> str:
        cog = center_of_gravity(self.points)
        diffs_from_cog = [round(distance_between_pts(cog, pt),8) for pt in self.points]
        min_diff = min(diffs_from_cog)
        pos_of_min = diffs_from_cog.index(min_diff)
        sorted_cog_diffs = reorder(diffs_from_cog, pos_of_min)
        ss = ".".join([f"{str(round(d,5))}" for d in sorted_cog_diffs])
        return ss

    # is_same_block() returns true if the blocks b1 and b2 have exactly the same size and shape
    @staticmethod
    def is_same_block2(b1: Blok, b2: Blok) -> bool:
        b1_id, b1_id2 = b1.gen_id2()

        e1 = Blok.find_flip_edge(b1)
        b1_flip = Blok.flip(b1, e1)

        b1_flip_id, b1_flip_id2 = b1_flip.gen_id2()

        b2_id, b2_id2 = b2.gen_id2()

        return (b1_id == b2_id) or (b1_id2 == b2_id) or (b1_flip_id == b2_id) or (b1_flip_id2 == b2_id)

    def gen_id2(self) -> str:
        cog = center_of_gravity(self.points)

        diffs_from_cog = [distance_between_pts(cog, pt) for pt in self.points]
        min_diff = min(diffs_from_cog)

        pos_of_min = diffs_from_cog.index(min_diff)
        sorted_cog_diffs = reorder(diffs_from_cog, pos_of_min)

        sorted_cog_diffs_rev = reorder(diffs_from_cog, pos_of_min, reverse=True)

        ss = ".".join([f"{str(round(d,5))}" for d in sorted_cog_diffs])
        ss_rev = ".".join([f"{str(round(d,5))}" for d in sorted_cog_diffs_rev])

        return (ss, ss_rev)

    def gen_id(self) -> Tuple[int, float]:
        """
        ID will be the pair.  First part is the number of edge, second part
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
    
    def get_bounding_box(self):
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


    # merge() will return the block formed after merging blocks b1 and b2
    # b1 and b2 are retained in the resulting block as component blocks
    @staticmethod
    def merge(b1: Blok, b2: Blok) -> Blok:
        only_b1, shared, only_b2 = find_shared_pts(b1.points, b2.points)
        bp1:List[Point] = b1.points
        bp2:List[Point] = b2.points
        i1 = 0
        i2 = 0
        if len(only_b1) == 0 or len(only_b2) == 0:
            #raise  Exception("neither only_b1 nor only_b2 should be zero")
            pass

        merged_points:List[Point] = []

        point_only_in_b1 = next(iter(only_b1))
        # find the index of a point in bp1 that is not in shared
        i1 = bp1.index(point_only_in_b1)

        start = bp1[i1]
        # now march along the b1 points until we hit the shared points
        # each point we traverse becomes part of the newly merged block
        while get_item(bp1, i1) not in shared:
            merged_points.append(get_item(bp1, i1))
            i1 = i1+1

        # now that we reached a point that is shared, switch over to traversing over
        # the b2 points
        # first find the index of the point in bp2 equal to the shared point that we
        # just encountered
        i2 = bp2.index(get_item(bp1, i1))
        # add it to the new block
        merged_points.append(get_item(bp2, i2))
        i2 = i2 + 1

        # now march along the b2 points until we hit the shared points
        # each point we traverse becomes part of the newly merged block
        while get_item(bp2, i2) not in shared:
            merged_points.append(get_item(bp2, i2))
            i2 = i2+1

        # now that we again reached a point that is shared, switch over to traversing over
        # the b1 points
        # first find the index of the point in bp1 equal to the shared point that we
        # just encountered
        i1 = bp1.index(get_item(bp2, i2))
        # add it to the new block
        merged_points.append(get_item(bp1, i1))
        i1 = i1 + 1

        # now finish going through bp1 until we get back to the start point
        while get_item(bp1, i1) != start:
            merged_points.append(get_item(bp1, i1))
            i1 = i1+1

        return Blok(merged_points)

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


def gen_next_level(levelset, b1, not_added=None):
    unique_blocks = set()
    cntr = 0
    for b0 in iter(levelset):
        for b0e in range(Blok.num_edges(b0)):
            for b1e in range(Blok.num_edges(b1)):
                (b,bm) = Blok.align_blocks_on_edge(b0, b0e, b1, b1e)
                newb = Blok.merge(b, bm)
                flip_edge = Blok.find_flip_edge(newb)
                flipped_newb = Blok.flip(newb, flip_edge)
                cntr = cntr + 1
                #print(f"iteration: {cntr}")
                newb_not_in = newb not in unique_blocks
                flipped_newb_not_in = flipped_newb not in unique_blocks
                if newb_not_in and flipped_newb_not_in:
                    xnewb_not_in = newb not in unique_blocks
                    xflipped_newb_not_in = flipped_newb not in unique_blocks
                    newbid = newb.gen_id()
                    flipped_newbid = flipped_newb.gen_id()
                    if newb.gen_id() != flipped_newb.gen_id():
                        # print(f"newb: {newb}")
                        # print(f"flipped_newb: {flipped_newb}")
                        # print(f"flipped_edge: {str(flip_edge)}")
                        capture_bloks(newb, flipped_newb)
                if newb_not_in and flipped_newb_not_in:
                    print(f"adding newb: {newb}: id:{newb.gen_id()}, flipped_newb_id: {flipped_newb.gen_id()}")
                    unique_blocks.add(newb)
                else:
                    if not_added != None:
                        not_added.append(newb)
    return unique_blocks

def capture_bloks(b1:Blok, b2:Blok, filename="debug1.svg"):
    cv = Canvas(width=20, height=20, nrows=3, ncols=3)
    sh1 = PolygonShape(b1.points, style=Style(color='black'))
    sh2 = PolygonShape(b2.points, style=Style(color='red'))
    cv = Canvas.add_shape2(cv, sh1, cv.get_box_for_cell(0, 0), label=f"#blocks: {str(b1.gen_id())}")
    cv = Canvas.add_shape2(cv, sh2, cv.get_box_for_cell(1, 1), label=f"#blocks: {str(b2.gen_id())}")
    with open(filename, 'w') as fd:
            Canvas.render_as_svg(cv, file=fd)

def runn(basic, num_levels, svg_file):
    b0 = Blok(basic)
    b1 = Blok.rotate(b0, Blok.get_edge(b0,0).start_pt, math.pi/4)
    prev_level = set()
    prev_level.add(b0)
    all = [(1,b0)]

    for lev in range(num_levels-1):
        curr_level = gen_next_level(prev_level, b1)
        for s in iter(curr_level):
            #print(s)
            all.append((lev+2,s))
        prev_level = curr_level

    normalized_bloks = [(n, Blok.normalize(b)) for n,b in all]
    bigbox = Box.bounding_box_of_boxex([PolygonShape(b.points).bounding_box() for _,b in normalized_bloks])

    d = int(math.sqrt(len(all)))
    cv = Canvas(width=20, height=20, nrows=d+2, ncols=d+2)
    i = 0
    for n,b in normalized_bloks:
        c = int(i / d)
        r = int(i % d)
        sh = PolygonShape(b.points, style=Style(color='black'))
        box = cv.get_box_for_cell(r, c)
        cv = Canvas.add_shape3(cv, sh, box, bigbox, label=f"#blocks: {str(n)}")
        # print(f"{i} ==============================")
        # print(f"bloc {n}: {b}")
        # print(f"bloc {n} ID:: {b.gen_id()}")
        i = i + 1
    print(f"num shapes: {len(all)}")
    with open(svg_file, 'w') as fd:
            Canvas.render_as_svg(cv, file=fd)

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

def search_if_same_circular_lists(s1, s2):
    c1 = Circular(s1,0)
    for idx in range(len(s2)):
        c2 = Circular(s2, idx)
        if equal_sequence(iter(c1), iter(c2)):
            return True
    return False

def equal_sequence(it1, it2) -> bool:
    return tuple(it1) == tuple(it2)

if __name__ == '__main__':

    SQUARE = [Point(0.0,0.0), Point(1.0,0.0), Point(1.0,1.0), Point(0.0,1.0)]    
    runn(SQUARE, 4, "square4.svg")

    SQUARE = [Point(0.0,0.0), Point(1.0,0.0), Point(1.0,1.0), Point(0.0,1.0)]    
    runn(SQUARE, 6, "square6.svg")

    altitude= (1/2)*math.sqrt(3)
    TRIANGLE = [Point(0.0, 0.0), Point(0.5,altitude),Point(1.0,0.0)]
    runn(TRIANGLE, 7, "triangle7.svg")

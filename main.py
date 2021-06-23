from __future__ import annotations
from typing import Set
from blokus import Blok, OverlapException
from blokus import UnequalEdgesException

from canvas import Canvas, CompositeShape, PolygonShape, Style, Box
from point import Point
import math
import time

def gen_next_level(levelset, b1, is_symmetric=False) -> Set[Blok]:
    unique_blocks = set()
    cntr = 0
    # if the piece we are adding is symmetric, namely all edges are indistinguiable from one another,
    # then we only need to attach one edge, there is no need to try all the edges as it would be the same
    if is_symmetric:
        b1_edges = [0]
    else:
        b1_edges = range(Blok.num_edges(b1))
    for b0 in iter(levelset):
        for b0e in range(Blok.num_edges(b0)):
            for b1e in b1_edges:
                try:
                    (b,bm) = Blok.align_blocks_on_edge(b0, b0e, b1, b1e)
                    newb = Blok.merge(b, bm)
                    flip_edge = Blok.find_flip_edge(newb)
                    flipped_newb = Blok.flip(newb, flip_edge)
                    cntr = cntr + 1
                    newb_not_in = newb not in unique_blocks
                    flipped_newb_not_in = flipped_newb not in unique_blocks
                    if newb_not_in and flipped_newb_not_in:
                        # print(f"adding newb: {newb}: id:{newb.gen_id()}, flipped_newb_id: {flipped_newb.gen_id()}")
                        # print(f"newb: {newb}")
                        unique_blocks.add(newb)
                except OverlapException:
                    continue
                except UnequalEdgesException:
                    continue
    return unique_blocks

levels_cache = {}

def runn(name, basic, num_levels, svg_file, is_symmetric = False):
    start_time = time.time()
    b0 = Blok(basic)
    b1 = Blok.rotate(b0, Blok.get_edge(b0,0).start_pt, math.pi/4)
    prev_level = set()
    prev_level.add(b0)
    all = [(1,b0)]

    for lev in range(num_levels-1):
        cache = levels_cache.get((name,lev), None)
        if cache:
            curr_level = cache
        else:
            curr_level = gen_next_level(prev_level, b1, is_symmetric)
            levels_cache[(name,lev)] = curr_level
        for s in iter(curr_level):
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
        box = cv.get_box_for_cell(r, c)
        sh = PolygonShape(b.points, style=Style(color='black', size='0.05'))
        if b.component_blocks:
            component_polygons = [PolygonShape(bp.points, style=Style(color='red')) for bp in b.component_blocks] + [sh]
            sh = CompositeShape(component_polygons)
        cv = Canvas.add_shape3(cv, sh, box, bigbox, margin_percent=0.3, label=f"#blocks: {str(n)}")
        i = i + 1

    with open(svg_file, 'w') as fd:
            Canvas.render_as_svg(cv, file=fd)
    print(f"{name}: Num shapes: {len(all)} --- Time: {(time.time() - start_time)} seconds ---")

def pentagon(r):
    sum_interior_angles = (5-2)*math.pi
    pent_angle = ((sum_interior_angles)/5)/2
    ND = r * math.cos(pent_angle)
    ON = r * math.sin(pent_angle)
    BC = 2 * ND
    FC = BC * math.sin(pent_angle)
    BF = BC * math.cos(pent_angle)
    FO = r -  BF

    A = (-FC,FO)
    B = (0.0, r)
    C = (FC, FO)
    D = (ND, -ON)
    E = (-ND, -ON)
    return [A,B,C,D,E]

if __name__ == '__main__':

    SQUARE = [Point(0.0,0.0), Point(1.0,0.0), Point(1.0,1.0), Point(0.0,1.0)]    
    TRIANGLE = [Point(0.0, 0.0), Point(0.5,(1/2)*math.sqrt(3)),Point(1.0,0.0)]
    ISOTRIANGLE = [Point(0.0, 0.0), Point(1.0, 1.0), Point(1.0,0.0)]
    PENTAGON = [Point(x,y) for (x,y) in pentagon(1.0)]

    h = math.sin(math.pi/3.0)
    hex = [ (-0.5,-h), (-1,0), (-0.5,h), (0.5,h), (1,0), (0.5,-h) ]
    HEXAGON = [Point(x,y) for (x,y) in hex]

    is_symmetric = True
    
    start_time = time.time()
    for i in range(1,8):
        loop_time = time.time()
        print(f"\tStart LOOP {i}")
        runn("\t\tTRIANGLE", TRIANGLE, i, f"svg/triangle{i}.svg", is_symmetric)
        runn("\t\tSQUARE", SQUARE, i, f"svg/square{i}.svg", is_symmetric)
        runn("\t\tPENTAGON", PENTAGON, i, f"svg/pentagon{i}.svg", is_symmetric)
        runn("\t\tHEXAGON", HEXAGON, i, f"svg/hexagon{i}.svg", is_symmetric)
        # runn("ISOTRIANGLE", ISOTRIANGLE, i, f"svg/iso_triangle{i}.svg")
        print(f"\tEnd LOOP {i} --- Time: {(time.time() - loop_time)} seconds ---")
        pass
    print(f"TOTAL --- Time: {(time.time() - start_time)} seconds ---")


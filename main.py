from __future__ import annotations
from blokus import Blok
from canvas import Canvas, PolygonShape, Style, Box
from point import Point
import math

def capture_bloks(b1:Blok, b2:Blok, filename="debug1.svg"):
    cv = Canvas(width=20, height=20, nrows=3, ncols=3)
    sh1 = PolygonShape(b1.points, style=Style(color='black'))
    sh2 = PolygonShape(b2.points, style=Style(color='red'))
    cv = Canvas.add_shape2(cv, sh1, cv.get_box_for_cell(0, 0), label=f"#blocks: {str(b1.gen_id())}")
    cv = Canvas.add_shape2(cv, sh2, cv.get_box_for_cell(1, 1), label=f"#blocks: {str(b2.gen_id())}")
    with open(filename, 'w') as fd:
            Canvas.render_as_svg(cv, file=fd)

def gen_next_level(levelset, b1) -> Set[Blok]:
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
                    print(f"adding newb: {newb}: id:{newb.gen_id()}, flipped_newb_id: {flipped_newb.gen_id()}")
                    unique_blocks.add(newb)
    return unique_blocks


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
        i = i + 1
    print(f"num shapes: {len(all)}")
    with open(svg_file, 'w') as fd:
            Canvas.render_as_svg(cv, file=fd)

if __name__ == '__main__':

    SQUARE = [Point(0.0,0.0), Point(1.0,0.0), Point(1.0,1.0), Point(0.0,1.0)]    
    runn(SQUARE, 4, "square4.svg")

    SQUARE = [Point(0.0,0.0), Point(1.0,0.0), Point(1.0,1.0), Point(0.0,1.0)]    
    runn(SQUARE, 6, "square6.svg")

    altitude= (1/2)*math.sqrt(3)
    TRIANGLE = [Point(0.0, 0.0), Point(0.5,altitude),Point(1.0,0.0)]
    runn(TRIANGLE, 7, "triangle7.svg")

import unittest
from blokus import Blok
from point import Point, is_path_clockwise, rearrange_origin
from edge import Edge
import numpy as np
import math
from canvas import Canvas, Shape, Line, Style, Box, PolygonShape, CompositeShape

SQUARE = [Point(0.0,0.0), Point(1.0,0.0), Point(1.0,1.0), Point(0.0,1.0)]
NP_SQUARE = np.array([[0,0], [1,0], [1,1], [0,1]])

SRH=math.sqrt(1/2)
ANGLED_SQUARE = np.array([[0,0],[SRH,SRH],[0,2*SRH],[-SRH,SRH]])


A = [Point(0,0), Point(1,0), Point(1,1), Point(0,1)]
B = [Point(0,0), Point(SRH,SRH), Point(0,2*SRH), Point(-SRH,SRH), Point(3,3)]
C = [Point(SRH,SRH), Point(0,2*SRH), Point(-SRH,SRH), Point(0,0), Point(-3,3)]
D = [Point(-SRH,SRH), Point(0,0), Point(SRH,SRH), Point(0,2*SRH), Point(3,-3)]
E = [Point(0,2*SRH), Point(-SRH,SRH), Point(0,0), Point(SRH,SRH), Point(-3,-3)]

INC=[Point(6,6)]

class TestRender(unittest.TestCase):
    
    def setUp(self) -> None:
        return super().setUp()

    def test_canvas_shapes(self):
        b0 = Blok([Point(10,11), Point(12,13), Point(13,11)]) 
        nb0 = Blok.normalize(b0)
        c1 = Canvas(width=20, height=20)
        shape_b0 = PolygonShape(b0.points)
        shape_nb0 = PolygonShape(nb0.points, style=Style(color='red'))
        bbox = shape_nb0.bounding_box()
        path = bbox.get_path()
        shape_bb = PolygonShape(path, style=Style(color='blue'))

        c2 = Canvas.add_shape(c1, shape_b0)
        c3 = Canvas.add_shape(c2, shape_nb0)
        c4 = Canvas.add_shape(c3, shape_bb)

        box = Box(Point(10,0), Point(20,10))
        c5 = Canvas.add_shape2(c4, PolygonShape(nb0.points, style=Style(color='green')), box)
        box2 = Box(Point(10,10), Point(20,20))
        c6 = Canvas.add_shape2(c5, PolygonShape(nb0.points, style=Style(color='yellow')), box2)
        
        with open('canvas1.svg', 'w') as fd:
            Canvas.render_as_svg(c6, file=fd)

    def test_canvas_shapes2(self):
        b0 = Blok([Point(10,11), Point(12,13), Point(13,11)]) 
        nb0 = Blok.normalize(b0)

        c1 = Canvas(width=20, height=20)
        shape_b0 = PolygonShape(b0.points, style=Style(color='green'))
        shape_nb0 = PolygonShape(nb0.points, style=Style(color='red'))

        shape_c1 = CompositeShape([shape_b0, shape_nb0])

        c1 = Canvas.add_shape(c1, shape_c1)

        box = Box(Point(10,0), Point(20,10))
        
        c2 = Canvas.add_shape2(c1, shape_c1, box)
        c3 = Canvas.add_shape2(c2, shape_c1, Box(Point(10,10), Point(12,12)))
        
        with open('canvas2.1.svg', 'w') as fd:
            Canvas.render_as_svg(c3, file=fd)

    def test_canvas_shapes3(self):
        b0 = Blok([Point(10,11), Point(12,13), Point(13,11)]) 
        nb0 = Blok.normalize(b0)
        shape_b0 = PolygonShape(b0.points, style=Style(color='green'))
        shape_nb0 = PolygonShape(nb0.points, style=Style(color='red'))
        shape_c1 = CompositeShape([shape_b0, shape_nb0])

        cv = Canvas(width=20, height=20, nrows=4, ncols=4)

        for r in range(4):
            for c in range(4):
                box = cv.get_box_for_cell(r, c)
                cv = Canvas.add_shape2(cv, shape_c1, box)

        with open('canvas3.svg', 'w') as fd:
            Canvas.render_as_svg(cv, file=fd)

    def test_canvas_shapes4(self):
        
        b0 = Blok([Point(10,11), Point(12,13), Point(13,11)]) 
        nb0 = Blok([Point(5,5), Point(5,8), Point(8,9), Point(7,5)]) 
        b1 = Blok([Point(10,11), Point(12,13), Point(13,11)]) 
        shape_b0 = PolygonShape(b0.points, style=Style(color='green'))
        shape_b1 = PolygonShape(b1.points, style=Style(color='blue'))
        shape_nb0 = PolygonShape(nb0.points, style=Style(color='red'))
        shape_c1 = CompositeShape([shape_b0, shape_nb0])

        cv = Canvas(width=20, height=20, nrows=4, ncols=4)

        for r in range(4):
            for c in range(4):
                box = cv.get_box_for_cell(r, c)
                if r == 3 and c == 3:
                    cv = Canvas.add_shape2(cv, shape_b1, box, label=f"({str(r)},{str(c)})")
                else:
                    cv = Canvas.add_shape2(cv, shape_c1, box, label=f"({str(r)},{str(c)})")

        with open('canvas4.svg', 'w') as fd:
                Canvas.render_as_svg(cv, file=fd)

    def test_simple_generator(self):
        b0 = Blok(SQUARE)
        b1 = Blok(SQUARE)
        cv = Canvas(width=20, height=20, nrows=4, ncols=4)

        for r in range(4):
            for c in range(4):
                (_,b2) = Blok.align_blocks_on_edge(b0, r, b1, c)
                b0_shape = PolygonShape(b0.points, style=Style(color='blue'))
                b2_shape = PolygonShape(b2.points, style=Style(color='red'))
                b0_b2_shape = CompositeShape([b0_shape, b2_shape])
                box = cv.get_box_for_cell(r, c)
                cv = Canvas.add_shape2(cv, b0_b2_shape, box, label=f"({str(r)},{str(c)})")

        with open('canvas5.svg', 'w') as fd:
                Canvas.render_as_svg(cv, file=fd)


    def test_merge(self):
        b0 = Blok(SQUARE)
        b1 = Blok(SQUARE)
        (b0,b2) = Blok.align_blocks_on_edge(b0, 0, b1, 0)
        mb = Blok.merge(b0, b2)


    def test_rotate(self):
        b0 = Blok([Point(1,1), Point(2,3), Point(3,1)]) 
        br = Blok.rotate(b0, Point(1,1),  math.pi* ( 1.0 / 2.0 ))

        self.assertEqual(br, Blok([Point(1,1), Point(3,0), Point(1,-1)]))
        br2 = Blok.rotate(b0, Point(1,1),  math.pi)

        self.assertEqual(br2, Blok([Point(1,1), Point(0,-1), Point(-1,1)]))
        br3 = Blok.rotate(b0, Point(1,1),  -math.pi)                
        self.assertEqual(br2, br3)

    def test_flip(self):
        b0 = Blok([Point(1,1), Point(2,3), Point(3,1)]) 
        bf = Blok.flip(b0, Blok.get_edge(b0, 2))
        print(bf)
        self.assertEqual(bf, Blok([Point(3,1), Point(2,-1), Point(1,1)]))

    def test_determinent(self):
        a = Point(0,0)
        b = Point(1,0)
        c = Point(1,1)
        d = Point(0,1)

        print(is_path_clockwise(c,b,a))
        print(is_path_clockwise(a,b,c))
        print(is_path_clockwise(a,d,c))
        print(is_path_clockwise(c,d,a))
        print(is_path_clockwise(b,a,c))
        print(is_path_clockwise(c,a,b))        

    def test_rearrange(self):
        points = [Point(0,0), Point(0,2), Point(2,2), Point(2,0)]
        print(f"orig: {points}")
        for i in range(len(points)):
            pt = points[i]
            rearr = rearrange_origin(points, pt)
            print(f"rearr({i}): {rearr}")

    def test_find_conv_hull(self):
        pass

    def test_align_block1(self):
        b1 = Blok([Point(0,0), Point(0,2), Point(2,2), Point(2,0)])
        b2 = Blok.translate(b1, 4, 4)
        b3 = Blok.rotate(b2, b2.points[0], math.pi/4)

        print(f"b1: {b1}")
        print(f"b3: {b3}")

        for i in range(4):
            print(f"{i}:")
            for j in range(4):
                orig, aligned = Blok.align_blocks_on_edge(b1, i, b3, j)
                print(f"aligned edge {i} {j}: {aligned}")

if __name__ == '__main__':
    unittest.main()

    
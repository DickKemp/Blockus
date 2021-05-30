from __future__ import annotations

from typing import List, Set, Tuple 
import numpy as np
import math
import sys

from point import Point, calc_angle, rearrange_origin, is_path_clockwise, is_eq_float, get_item, find_shared_pts, diff
from edge import Edge

class Style():
    def __init__(self, color='black', size=None, dashed=None):
        self.color = color
        self.size = size
        self.dashed = dashed

class Shape():
    def __init__(self):
        raise Exception("not implemented")
    def bounding_box(self: Shape) -> Box:
        raise Exception("not implemented")
    def scale(self: Shape, xfactor: float, yfactor: float) -> Shape:
        raise Exception("not implemented")
    def translate(self: Shape, pt: Point) -> Shape:
        raise Exception("not implemented")
    def get_midpoint(self: Shape) -> Point:
        raise Exception("not implemented")        
    
    @staticmethod
    def set_label(s:Shape, label:str) -> Shape:
        s2 = s.copy()
        s2.label = label
        return s2
    
class CompositeShape(Shape):
    def __init__(self: CompositeShape, shapes:List[Shape], label:str=None, style:Style=Style()) -> None:
        self.shapes = shapes
        self.label = label
        self.style = style

    def bounding_box(self: CompositeShape) -> Box:
        bbs = [s.bounding_box() for s in self.shapes]
        minx = min([b.minx() for b in bbs])
        miny = min([b.miny() for b in bbs])
        maxx = max([b.maxx() for b in bbs])
        maxy = max([b.maxy() for b in bbs])
        return (Box(Point(minx, miny), Point(maxx, maxy)))
    
    def scale(self: CompositeShape, xfactor: float, yfactor: float) -> CompositeShape:
        scaled_shapes = [s.scale(xfactor, yfactor) for s in self.shapes]
        return CompositeShape(scaled_shapes, self.label, self.style)

    def translate(self: CompositeShape, pt: Point) -> CompositeShape:
        translated_shapes = [s.translate(pt) for s in self.shapes]
        return CompositeShape(translated_shapes, self.label, self.style)

    def get_midpoint(self: CompositeShape) -> Point:
        x = sum([s.get_midpoint().x for s in self.shapes])/len(self.shapes)
        y = sum([s.get_midpoint().y for s in self.shapes])/len(self.shapes)
        return Point(x,y)

    def copy(self):
        return CompositeShape(self.shapes, self.label, self.style)

class PolygonShape(Shape):
    def __init__(self: PolygonShape, points:List[Point], label:str=None, style:Style=Style()) -> None:
        self.points = points
        self.label = label
        self.style = style
    
    def bounding_box(self: PolygonShape) -> Box:
        minx = min([p.x for p in self.points])
        miny = min([p.y for p in self.points])
        maxx = max([p.x for p in self.points])
        maxy = max([p.y for p in self.points])
        return (Box(Point(minx, miny), Point(maxx, maxy)))

    def scale(self: PolygonShape, xfactor: float, yfactor: float) -> PolygonShape:
        scaled = [Point(p.x * xfactor, p.y * yfactor) for p in self.points]
        return PolygonShape(scaled, self.label, self.style)

    def translate(self: PolygonShape, pt: Point) -> PolygonShape:
        translated = [Point(p.x + pt.x, p.y + pt.y) for p in self.points]
        return PolygonShape(translated, self.label, self.style)        

    def get_midpoint(self: PolygonShape) -> Point:
        x = sum([p.x for p in self.points])/len(self.points)
        y = sum([p.y for p in self.points])/len(self.points)
        return Point(x,y)

    def copy(self):
        return PolygonShape(self.points, self.label, self.style)

class Line():
    def __init__(self: Line, point1: Point, point2: Point, label:str=None, style:Style=None) -> None:
        self.point1 = point1
        self.point2 = point2
        self.label = label
        self.style = style

class Box():
    def __init__(self: Box, lower_left_pt: Point, upper_right_pt: Point) -> None:
        self.lower_left_pt = lower_left_pt
        self.upper_right_pt = upper_right_pt
    def get_path(self):
        minx = self.minx()
        miny = self.miny()
        maxx = self.maxx()
        maxy = self.maxy()
        return [Point(minx,miny), Point(minx, maxy), Point(maxx, maxy), Point(maxx,miny)]
    def width(self):
        return self.maxx() - self.minx()
    def height(self):
        return self.maxy() - self.miny()
    def minx(self):
        return self.lower_left_pt.x
    def miny(self):
        return self.lower_left_pt.y
    def maxx(self):
        return self.upper_right_pt.x
    def maxy(self):
        return self.upper_right_pt.y
    
    @staticmethod
    def bounding_box_of_boxex(boxes:List[Box]) -> Box:
        minx = min([b.minx() for b in boxes])
        miny = min([b.miny() for b in boxes])
        maxx = max([b.maxx() for b in boxes])
        maxy = max([b.maxy() for b in boxes])
        return Box(Point(minx, miny), Point(maxx, maxy))

# ##################################################
#
# Canvas is a place where you can place blocks, then later you can render the block

class Canvas():
    def __init__(self, shapes = list(), lines=list(), height=10, width=10, nrows=1, ncols=1) -> None:
        self.shapes = shapes
        self.lines = lines
        self.viewBoxHeight = height
        self.viewBoxWidth = width
        self.num_rows = nrows
        self.num_cols = ncols

    def get_box_for_cell(self, cell_row_, cell_col_):
        rhgt = self.viewBoxHeight/self.num_rows
        cwdth = self.viewBoxWidth/self.num_cols
        # cell_row = cell_row_
        cell_col = cell_col_
        cell_row = self.num_rows - cell_row_ -1
        #cell_col = self.num_cols - cell_col_ -1
        return Box(Point(cwdth*cell_col, rhgt*cell_row), Point(cwdth*(cell_col+1), rhgt*(cell_row+1)))

    # add_shape will add a shape to the canvas c, where the center of the blocks is positioned
    # at the specified center point; the block will be styled as specified in style s
    @staticmethod
    def add_shape(c: Canvas, shape: Shape) -> Canvas:
        nc = Canvas(c.shapes + [shape], c.lines, c.viewBoxHeight, c.viewBoxWidth, c.num_rows, c.num_cols)
        return nc

    @staticmethod
    def add_shape2(c: Canvas, shape: Shape, container_box: Box, outline:bool=True, margin_percent:float=0.2, label:str=None) -> Canvas:
        shape_box = shape.bounding_box()
        width_margin = container_box.width() * margin_percent
        height_margin = container_box.height() * margin_percent
        inner_box = Box(Point(container_box.lower_left_pt.x + width_margin/2, container_box.lower_left_pt.y + height_margin/2), 
                        Point(container_box.upper_right_pt.x - height_margin/2, container_box.upper_right_pt.y - height_margin/2))

        scalex = inner_box.width() / (shape_box.width())
        scaley = inner_box.height() / (shape_box.height())

        scalexy = min([scalex, scaley])

        scaled_shape = shape.scale(xfactor=scalexy, yfactor=scalexy)
        scaled_shape_box = scaled_shape.bounding_box()

        translation_amt = diff(inner_box.lower_left_pt, scaled_shape_box.lower_left_pt)
        translated_shape = scaled_shape.translate(translation_amt)
        outline_shape = []
        if outline:
            outline_shape.append(PolygonShape(container_box.get_path(), label=label))
        nc = Canvas(c.shapes + [translated_shape] + outline_shape, c.lines, c.viewBoxHeight, c.viewBoxWidth, c.num_rows, c.num_cols)
        return nc

    @staticmethod
    def add_shape3(c: Canvas, shape: Shape, container_box: Box, super_shape_box: Box, outline:bool=True, margin_percent:float=0.2, label:str=None) -> Canvas:
        shape_box = super_shape_box
        width_margin = container_box.width() * margin_percent
        height_margin = container_box.height() * margin_percent
        inner_box = Box(Point(container_box.lower_left_pt.x + width_margin/2, container_box.lower_left_pt.y + height_margin/2), 
                        Point(container_box.upper_right_pt.x - height_margin/2, container_box.upper_right_pt.y - height_margin/2))

        scalex = inner_box.width() / (shape_box.width())
        scaley = inner_box.height() / (shape_box.height())

        scalexy = min([scalex, scaley])

        scaled_shape = shape.scale(xfactor=scalexy, yfactor=scalexy)
        scaled_shape_box = scaled_shape.bounding_box()

        translation_amt = diff(inner_box.lower_left_pt, scaled_shape_box.lower_left_pt)
        translated_shape = scaled_shape.translate(translation_amt)
        outline_shape = []
        if outline:
            outline_shape.append(PolygonShape(container_box.get_path(), label=label))
        nc = Canvas(c.shapes + [translated_shape] + outline_shape, c.lines, c.viewBoxHeight, c.viewBoxWidth, c.num_rows, c.num_cols)
        return nc


    # add_line_to_canvas() will add a line from p1 to p2 to the canvas c
    @staticmethod
    def add_line(c: Canvas, line: Line) -> Canvas:
        nc = Canvas(c.shapes, c.lines + [line], c.viewBoxHeight, c.viewBoxWidth, c.num_rows, c.num_cols)
        return nc

    # render_svg() will render a canvas as an Svg
    @staticmethod
    def render_as_svg(c: Canvas, file=sys.stdout) -> None:
        print(f'<svg width="800" height="800" viewBox="0 0 {c.viewBoxWidth} {c.viewBoxHeight}" preserveAspectRatio="xMinYMin meet" xmlns="http://www.w3.org/2000/svg">', file=file)

        for s in c.shapes:
            Canvas.render_shape_as_svg(c, s, file)

        print('</svg>', file=file)        

    @staticmethod
    def render_shape_as_svg(c: Canvas, s: Shape, file=sys.stdout) -> None:
        if isinstance(s, PolygonShape):
            Canvas._draw_polygon_shape_svg(c, s, file=file)
        elif isinstance(s, CompositeShape):
            mid_pt = s.get_midpoint()
            if s.label:
                inv_pt = c.viewBoxHeight - mid_pt.y
                print(f'<text x="{mid_pt.x}" y="{inv_pt}" font-family="Verdana" font-size="0.35">{s.label}</text>', file=file)

            for sh in s.shapes:
                Canvas.render_shape_as_svg(c, sh, file)

    @staticmethod
    def _draw_polygon_shape_svg(c: Canvas, shape: Shape, file=sys.stdout) -> None:
        scale=1

        # label=None
        #if shape.label:
        #    label = f'<text x="{int(shape.points[2][0]*scale)}" y="{int(block.vertices[2][1]*scale)}">{title}</text>'
        #    print(label, file=file)

        print(f'<polygon points="0 0, 0 {c.viewBoxHeight}, {c.viewBoxWidth} {c.viewBoxHeight}, {c.viewBoxWidth} 0" fill="none" style="stroke:blue;stroke-width:0.1"/>', file=file)

        bb = shape.bounding_box()
        label_anchor = bb.lower_left_pt
        if shape.label:
            inv_pt = c.viewBoxHeight - label_anchor.y
            print(f'<text x="{label_anchor.x}" y="{inv_pt}" font-family="Verdana" font-size="0.35">{shape.label}</text>', file=file)

        points=""
        first=True
        for pt in shape.points:
            if first:
                sep=""
                first=False
            else:
                sep=", "
            inv_pt = c.viewBoxHeight - pt.y
            points=f"{points}{sep}{float(pt.x*scale)} {float(inv_pt*scale)}"

        print(f'<polygon points="{points}" fill="none" style="stroke:{shape.style.color};stroke-width:0.01"/>', file=file)

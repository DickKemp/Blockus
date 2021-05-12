from flask import Flask
from flask import request

import numpy as np
from blocks import Block, Edge, Point
from tests.block_test import A, B, C, D, E, INC, X1, X2
import math
import io

app = Flask(__name__)

@app.route("/")
def hello():
    id = 'rich'
    return f"Hello: {id}"

@app.route("/b")
def b1():
    a = np.array([[0,0], [1,0], [1,1], [0,1]]) + np.array([5,5])

    outstr = ""
    with io.StringIO(initial_value=None) as ofd:
        print('<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg">', file=ofd)
        ablock = Block(a)
        ablock.draw_svg(color='black', file=ofd)
        for i in np.linspace(0, (6/7)*2*math.pi, 6):

            ablock = Block(a)
            ablock.rotate2(ablock.vertices[0],  i)
            ablock.draw_svg(color='red', title=str(round(i,2)), file=ofd)

        print('</svg>', file=ofd)
        outstr = ofd.getvalue()
        ofd.close()
    return outstr

@app.route("/b2")
def b2():

    bx1 = Block(np.array(X1) + np.array([2,5]), False)
    bx2 = Block(np.array(X2) + np.array([4,3]), True)

    with io.StringIO(initial_value=None) as ofd:
        Block.draw_blocks([bx1, bx2], file=ofd)
        outstr = ofd.getvalue()
        ofd.close()
    return outstr


@app.route("/b3")
def b3():

    bx1 = Block(np.array(X1) + np.array([2,5]), False)
    bx2 = Block(np.array(X2) + np.array([4,3]), True)

    with io.StringIO(initial_value=None) as ofd:
        (moved, rot) = bx1.align_block_on_edge(4, bx2, 6)
        Block.draw_blocks([bx1,  rot], file=ofd)
        outstr = ofd.getvalue()
        ofd.close()
    return outstr
        

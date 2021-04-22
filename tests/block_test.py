import unittest
from blocks import Block, Edge, Point
import numpy as np
import math

SQUARE = [(0,0), (1,0), (1,1), (0,1)]
NP_SQUARE = np.array([[0,0], [1,0], [1,1], [0,1]])
SRH=math.sqrt(1/2)
ANGLED_SQUARE = np.array([[0,0],[SRH,SRH],[0,2*SRH],[-SRH,SRH]])

A = np.array([(0,0), (1,0), (1,1), (0,1)])
B = np.array([[0,0],[SRH,SRH],[0,2*SRH],[-SRH,SRH]]) + np.array([3,3])
C = np.array([[SRH,SRH],[0,2*SRH],[-SRH,SRH], [0,0]]) + np.array([-3,3])
D = np.array([[-SRH,SRH],[0,0],[SRH,SRH],[0,2*SRH]]) + np.array([3,-3])
E = np.array([[0,2*SRH],[-SRH,SRH], [0,0],[SRH,SRH]]) + np.array([-3,-3])

INC=np.array([6,6])

# +
X0 =   [[0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]]

# -
X1 =   [[0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
        [0.0, 4.0],
        [1.0, 4.0],
        [1.0, 3.0],
        [1.0, 2.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [3.0, 2.0],
        [3.0, 1.0],
        [3.0, 0.0],
        [2.0, 0.0],
        [1.0, 0.0]]

# +
X2 =   [[0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [3.0, 1.0],
        [3.0, 2.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0]]

# -
X3 =   [[0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [3.0, 0.0],
        [2.0, 0.0],
        [1.0, 0.0]]

class TestBlockMethods(unittest.TestCase):
    
    def setUp(self) -> None:
        self.block1 = Block(SQUARE)
        return super().setUp()

    # def test_circular_iterator(self):
    #     arr = ['a', 'b', 'c', 'd']
    #     counter = 0
    #     outarr = []
    #     for item in Block.circular_iter(arr):
    #         counter += 1
    #         outarr.append(item)
    #         if counter >= 10:
    #             break
    #     self.assertListEqual(['a','b','c','d','a','b','c','d','a','b'], outarr)

    def test_point_set1(self):
        points_a = {(0.0, 0.0), (1.0000001,0.0), (2.0,0.0), (3.0, 0.0), (4.0, 0.0), (5.0,0.0), (6.0,0.0), (7.0, 0.0)}
        points_b = {(0.9999999999,0.0), (3.0, 0.0), (5.0,0.0), (7.0, 0.0), (9.0, 0.0), (11.0, 0.0)}
        result_a_b = {(1.0,0.0), (3.0, 0.0), (5.0,0.0), (7.0, 0.0)}
        shared = Block.find_shared_pts(points_a, points_b)
        for i in shared:
            print(str(i))
        # print(str(shared))
        self.assertTrue(True)

    # def test_get_iter_between_items(self):
    #     arr = ['a', 'b', 'X', 'c', 'd', 'e', 'Y', 'f']
    #     print(list(Block.get_iter_between_items(arr, 'X', 'Y')))
    #     self.assertTrue(True)

    def test_t1(self):
        a = Block(A + INC, True)
        b = Block(B + INC, False)
        c = Block(C + INC, False)
        d = Block(D + INC, False)
        e = Block(E + INC, False)
        with open('test_1.svg', 'w') as ofd:
            # print('<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg">', file=ofd)
            Block.draw_blocks([a,b,c,d,e], file=ofd)
            # a.draw_svg(color='black', file=ofd)
            # b.draw_svg(color='red', file=ofd)
            # c.draw_svg(color='green', file=ofd)
            # d.draw_svg(color='orange', file=ofd)
            # e.draw_svg(color='brown', file=ofd)
            # print('</svg >', file=ofd)
        blks = [(b,0),(c,0),(d,0),(e,0)]
        for i in range(0,4):
            with open('test_align_{str(i)}.svg', 'w') as ofd:
                # print('<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg">', file=ofd)                
                blk = blks[i][0]
                edge = blks[i][1]
                (_,new_blk) = a.align_block_on_edge(0, blk, edge)
                Block.draw_blocks([new_blk],file=ofd)
                # print('</svg >', file=ofd)

        self.assertTrue(True)

    def test_get_num_edges(self):
        self.assertEqual(self.block1.num_edges(), 4)

    def test_get_edges(self):
        self.assertEqual(self.block1.get_edge(0), Edge((0,0), (1,0)))
        self.assertEqual(self.block1.get_edge(4), Edge((0,0), (1,0)))
        self.assertEqual(self.block1.get_edge(5), Edge((1,0), (1,1)))
        
    def test_cog(self):
        self.assertEqual(self.block1._center_of_gravity(), (0.5, 0.5))
    
    def test_find_angle(self):
        # calc_angle(vertex_pt, start_pt, end_pt)
        vertex_pt = np.array([2,1])
        start_pt = np.array([3,4])
        end_pt = np.array([4,-1])
        ang = Block.calc_angle(vertex_pt, start_pt, end_pt)
        print("angle")
        print(ang)
        print("angle 1")
        print(Block.calc_angle(np.array([5,2]), np.array([6,3]), np.array([4.35796048, 0.73992649])))
        print("angle 2")
        print(Block.calc_angle(np.array([5,2]), np.array([6,3]), np.array([3.73992649, 1.35796048])))
        print("angle 3")
        print(Block.calc_angle(np.array([5,2]), np.array([3.73992649, 1.35796048]), np.array([6,3])))

    def test_move(self):
        a = np.array([[0,0], [1,0], [1,1], [0,1]]) + np.array([5,5])
        ablock = Block(a)
        print("")
        print(ablock)
        moved_block = ablock.move(np.array([5,5]), np.array([3,3]))
        print(moved_block)
        self.assertTrue(True)

    def test_flip(self):
        a = np.array([[0,0], [1,0], [1,1], [0,1]]) + np.array([2,2])
        ablock = Block(a)
        print("")
        print(ablock)
        flipped_block = ablock.flip(ablock.get_edge(0))
        print(flipped_block)
        self.assertTrue(True)

    def test_flip2(self):
        a = np.array([[0,0], [1,0], [1,1], [0,1]]) + np.array([5,0])
        ablock = Block(a)
        print("")
        print(ablock)
        flipped_block = ablock.flip2(1,0)
        print(flipped_block)
        self.assertTrue(True)

    def test_flip3(self):
        bx2 = Block(np.array(X2) + np.array([3,3]))
        
        with open('bx2.svg', 'w') as fd1:
            bx2.draw_svg(color='black', file=fd1, wrap=True)
        
        with open('bx2_flip.svg', 'w') as fd2:        
            flipped_bx2 = bx2.flip2(1,0)
            flipped_bx2.draw_svg(color='red', file=fd2, wrap=True)
        self.assertTrue(True)

    def test_move1(self):
        bx1 = Block(np.array(X1) + np.array([2,5]), False)
        bx2 = Block(np.array(X2) + np.array([4,3]), True)

        with open('init_move_1.svg', 'w') as fd1:
            Block.draw_blocks([bx1, bx2], file=fd1)

        with open('post_move_1.svg', 'w') as fd2:        
            (moved, rot) = bx1.align_block_on_edge(4, bx2, 6)
            Block.draw_blocks([bx1,  rot], file=fd2)

        self.assertTrue(True)

    def test_rotate1(self):
        a = np.array([[0,0], [1,0], [1,1], [0,1]]) + np.array([5,2])
        with open('s1.svg', 'w') as ofd:
            print('<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg">', file=ofd)

            ablock = Block(a)
            ablock.draw_svg(color='black', file=ofd)

            ablock.rotate2(ablock.vertices[0],  math.pi * 0.9)
            ablock.draw_svg(color='red', title="0.9pi", file=ofd)

            ablock = Block(a)
            ablock.rotate2(ablock.vertices[0], - math.pi * 0.9)
            ablock.draw_svg(color='green', title="-0.9pi", file=ofd)

            print('</svg>', file=ofd)

        self.assertTrue(True)        
    
    def test_rotate2(self):
        a = np.array([[0,0], [1,0], [1,1], [0,1]]) + np.array([5,5])
        with open('s2.svg', 'w') as ofd:
            print('<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg">', file=ofd)

            ablock = Block(a)
            ablock.draw_svg(color='black', file=ofd)
            for i in np.linspace(0, 2*math.pi, 20):

                ablock = Block(a)
                ablock.rotate2(ablock.vertices[0],  i)
                ablock.draw_svg(color='red', title=str(round(i,2)), file=ofd)

            print('</svg>', file=ofd)

        self.assertTrue(True)        

    def test_rotate3(self):
        a = np.array([[0,0], [1,0], [1,1], [0,1]]) + np.array([5,5])
        with open('s3.svg', 'w') as ofd:
            print('<svg width="1000" height="1000" xmlns="http://www.w3.org/2000/svg">', file=ofd)

            ablock = Block(a)
            ablock.draw_svg(color='black', file=ofd)
            for i in np.linspace(0, (6/7)*2*math.pi, 6):

                ablock = Block(a)
                ablock.rotate2(ablock.vertices[0],  i)
                ablock.draw_svg(color='red', title=str(round(i,2)), file=ofd)

            print('</svg>', file=ofd)

        self.assertTrue(True)        


if __name__ == '__main__':
    unittest.main()
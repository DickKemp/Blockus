import unittest
from blocks import Block
SQUARE = [(0,0), (1,0), (1,1), (0,1)]
class TestStringMethods(unittest.TestCase):
    
    def setUp(self) -> None:
        self.block1 = Block(SQUARE)
        return super().setUp()
    
    def test_get_num_edges(self):
        self.assertEqual(self.block1.get_num_edges(), 4)

    def test_get_edges(self):
        self.assertEqual(self.block1.get_edge_points(0), ((0,0), (1,0)))
        self.assertEqual(self.block1.get_edge_points(4), ((0,0), (1,0)))
        self.assertEqual(self.block1.get_edge_points(5), ((1,0), (1,1)))

    def test_cog(self):
        self.assertEqual(self.block1._center_of_gravity(), (0.5, 0.5))
        
    """  self.assertTrue('FOO'.isupper())
    self.assertFalse('Foo'.isupper())
    with self.assertRaises(TypeError):
        s.split(2) """

if __name__ == '__main__':
    unittest.main()
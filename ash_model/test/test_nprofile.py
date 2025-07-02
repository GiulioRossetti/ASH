import unittest

from ash_model import NProfile, ASH


class NProfileTestCase(unittest.TestCase):
    def test_profile(self):

        p = NProfile(node_id=1)
        p.add_attribute("name", "Giulio")
        p.add_attribute("age", 37)
        p.add_attribute("opinion", 1)

        self.assertEqual(p.has_attribute("name"), True)
        self.assertEqual(p.has_attribute("surname"), False)

        self.assertEqual(p.get_attribute("name"), "Giulio")
        self.assertDictEqual(
            p.get_attributes(), {"name": "Giulio", "age": 37, "opinion": 1}
        )

        p.add_attributes(surname="Rossetti", location="Pisa")
        self.assertEqual(p.has_attribute("surname"), True)
        self.assertEqual(p.has_attribute("location"), True)

    def test_compare(self):

        p = NProfile(node_id=1)
        p.add_attributes(age=20, opinion=1)

        p1 = NProfile(node_id=2)
        p1.add_attributes(age=19, opinion=1)

        self.assertEqual(p >= p1, True)
        self.assertEqual(p <= p1, False)
        self.assertEqual(p == p1, False)
        self.assertEqual(p != p1, True)

    def test_creation(self):
        p = NProfile(node_id=1, age=20, opinion=1)
        self.assertEqual(p.has_attribute("age"), True)
        self.assertEqual(p.has_attribute("opinion"), True)

    def test_statistic(self):
        p = NProfile(node_id=1, pippo="pippo")
        p.add_statistic("pippo", "mean", 3)
        self.assertDictEqual(p.get_statistic("pippo", "mean"), {"mean": 3})
        self.assertDictEqual(p.get_statistic("pippo"), {"mean": 3})
        self.assertEqual(p.has_statistic("pippo", "mean"), True)
        self.assertEqual(p.has_statistic("pippo", "max"), False)
        self.assertListEqual(p.attribute_computed_statistics("pippo"), ["mean"])

    def test_ash_profile(self):
        h = ASH()
        p = NProfile(node_id=1, age=20, opinion=1)
        h.add_node(1, start=0, attr_dict=p)
        p2 = NProfile(node_id=2, age=19, opinion=1)
        h.add_node(2, start=0, attr_dict=p2)
        
        self.assertEqual(h.get_node_attribute(1, "age"), 20)
        self.assertEqual(h.get_node_attribute(2, "age"), 19)
        self.assertEqual(h.get_node_attribute(1, "opinion"), 1)
        self.assertEqual(h.get_node_attribute(2, "opinion"), 1)

        
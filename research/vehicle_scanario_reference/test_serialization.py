import dataclasses, json, unittest
from extract_streams import plain
@dataclasses.dataclass
class Transform:
    x: float
class SerializationTests(unittest.TestCase):
    def test_nested_transform_sequence(self):
        value = [Transform(1.0), Transform(2.0)]
        self.assertEqual(json.loads(json.dumps(plain(value))), [{"x": 1.0}, {"x": 2.0}])
if __name__ == "__main__":
    unittest.main()

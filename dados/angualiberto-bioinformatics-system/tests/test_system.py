import unittest
from src.core.system import AnguaLibertoSystem

class TestAnguaLibertoSystem(unittest.TestCase):

    def setUp(self):
        self.system = AnguaLibertoSystem("Test Project")

    def test_initialization(self):
        self.assertEqual(self.system.project_name, "Test Project")
        self.assertIsNotNone(self.system.work_dir)
        self.assertEqual(self.system.author, "AnguaLiberto")
        self.assertEqual(self.system.version, "2.0.0")
        self.assertEqual(self.system.license, "MIT")

    def test_analysis_configs(self):
        self.assertIn("min_sequence_length", self.system.analysis_configs)
        self.assertIn("max_sequence_length", self.system.analysis_configs)
        self.assertIn("confidence_threshold", self.system.analysis_configs)
        self.assertIn("enable_cache", self.system.analysis_configs)

    def test_citations_initialization(self):
        citations = self.system.initialize_citations()
        self.assertGreater(len(citations), 0)
        self.assertIn("text", citations[0])
        self.assertIn("author", citations[0])
        self.assertIn("date", citations[0])

if __name__ == '__main__':
    unittest.main()
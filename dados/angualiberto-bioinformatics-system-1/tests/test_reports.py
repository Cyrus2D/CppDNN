import unittest
from src.reports.pdf_generator import create_pdf_report
from src.reports.html_generator import create_html_report

class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        self.test_data = {
            "title": "Test Report",
            "author": "AnguaLiberto",
            "content": "This is a test report content.",
            "date": "2024-01-01"
        }
        self.pdf_filename = "test_report.pdf"
        self.html_filename = "test_report.html"

    def test_pdf_report_generation(self):
        result = create_pdf_report(self.test_data, self.pdf_filename)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith('.pdf'))

    def test_html_report_generation(self):
        result = create_html_report(self.test_data, self.html_filename)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith('.html'))

if __name__ == '__main__':
    unittest.main()
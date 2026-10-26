import importlib.util
from pathlib import Path
import tempfile
import unittest

script = Path(__file__).resolve().parents[1] / "scripts" / "check_public_files.py"
spec = importlib.util.spec_from_file_location("publication", script)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    def test_generated_artifacts_and_embedded_credentials_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.engine").write_bytes(b"synthetic engine")
            (root / "README.md").write_text("ghp_" + "z" * 36)
            problems = publication.inspect(root)
            self.assertEqual(len(problems), 2)
            self.assertTrue(any("model.engine" in problem for problem in problems))
            self.assertFalse(any("ghp_" in problem for problem in problems))

    def test_ignored_data_is_still_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data" / "synthetic.jpg").write_bytes(b"synthetic")
            self.assertEqual(len(publication.inspect(root)), 1)


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from tempfile import NamedTemporaryFile

from src.notifications.token_registry import TokenRegistry


class TestTokenRegistry(unittest.TestCase):
    def test_add_and_persist(self):
        with NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        try:
            reg = TokenRegistry(path)
            reg.add("abc")
            reg.add("abc")
            reg.add("def")

            reg2 = TokenRegistry(path)
            self.assertEqual(set(reg2.get_all()), {"abc", "def"})
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()

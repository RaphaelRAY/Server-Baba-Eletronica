class TokenRegistry:
    """Store FCM tokens in a text file."""

    def __init__(self, path: str = "tokens.txt"):
        """Prepare registry at the given file path."""
        self.path = path
        self.tokens = set()
        self._load()

    def _load(self) -> None:
        """Load tokens from disk if available."""
        try:
            with open(self.path, "r") as fh:
                self.tokens = set(t.strip() for t in fh if t.strip())
        except FileNotFoundError:
            self.tokens = set()

    def _save(self) -> None:
        """Persist tokens to disk."""
        with open(self.path, "w") as fh:
            for t in sorted(self.tokens):
                fh.write(t + "\n")

    def add(self, token: str) -> None:
        """Add a token if new and save immediately."""
        if not token or token in self.tokens:
            return
        self.tokens.add(token)
        self._save()

    def get_all(self):
        """Return all stored tokens."""
        return list(self.tokens)

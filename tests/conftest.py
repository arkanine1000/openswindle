import os

# Mock mode before any openswindle import: tests never hit an LLM provider.
os.environ.setdefault("OPENSWINDLE_MOCK_LLM", "true")

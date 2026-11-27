from pathlib import Path

knowledge_base = Path("knowledge_base")

for file_path in knowledge_base.glob("*.txt"):
    print("Loading:", file_path.name)
import os
from pathlib import Path
from rag_utils import create_embeddings, save_faiss_index

DATA_DIR = Path("data")
OUTPUT_DIR = Path("faiss_index")
OUTPUT_DIR.mkdir(exist_ok=True)

INDEX_PATH = OUTPUT_DIR / "indecimal_index.faiss"
META_PATH = OUTPUT_DIR / "metadata.pkl"

def load_documents(data_dir):
    """Load text from all .md files."""
    docs, metadata = [], []
    for file in data_dir.glob("*.md"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(text)
            metadata.append({"filename": file.name, "text": text})
    return docs, metadata


def main():
    print("📄 Loading documents...")
    docs, metadata = load_documents(DATA_DIR)

    print("🧠 Creating embeddings...")
    embeddings = create_embeddings(docs)

    print("💾 Saving FAISS index and metadata...")
    save_faiss_index(embeddings, metadata, str(INDEX_PATH), str(META_PATH))

    print("✅ Index build complete!")


if __name__ == "__main__":
    main()

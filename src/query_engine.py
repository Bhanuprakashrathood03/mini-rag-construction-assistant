import os
import faiss
import pickle
import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer


class QueryEngine:
    def __init__(self, index_path, metadata_path, model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=3):
        """Initialize FAISS retriever and Ollama-based LLM."""
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("❌ Missing FAISS index or metadata file.")

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        self.embedder = SentenceTransformer(model_name)
        self.top_k = top_k

    def retrieve_context(self, query: str):
        """Retrieve relevant chunks for query."""
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, self.top_k)

        contexts = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                contexts.append(self.metadata[idx]["text"])
        return contexts

    def query_with_ollama(self, query: str, model="llama3"):
        """Query the local Ollama LLM using the retrieved context."""
        context = "\n\n".join(self.retrieve_context(query))

        if not context.strip():
            return "Information not available in provided documents."

        prompt = f"""
You are an assistant for Indecimal Construction.
Answer based only on the context below.
If the answer cannot be found, say "Information not available in provided documents."

Context:
{context}

Question: {query}
Answer:
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt},
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            full_reply = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    full_reply += data.get("response", "")

            return full_reply.strip() or "⚠️ No response from model."

        except requests.exceptions.ConnectionError:
            return "❌ Ollama is not running. Please start it using: `ollama serve`."
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

    # 🧠 NEW: Stream the response token-by-token for real-time display
    def stream_ollama_response(self, query: str, model="llama3"):
        """Stream the response token-by-token from Ollama with clean decoding."""
        context = "\n\n".join(self.retrieve_context(query))

        if not context.strip():
            yield "Information not available in provided documents."
            return

        prompt = f"""
You are an assistant for Indecimal Construction.
Answer based only on the context below.
If the answer cannot be found, say "Information not available in provided documents."

Context:
{context}

Question: {query}
Answer:
"""

        try:
            with requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
                stream=True,
                timeout=60,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")

                        # 🧹 Clean escaped unicode and newline characters
                        chunk = (
                            chunk.replace("\\n", "\n")
                                 .replace("\\u0026", "&")
                                 .replace("\\u2019", "'")
                                 .replace("\\u201c", '"')
                                 .replace("\\u201d", '"')
                        )
                        yield chunk

                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.ConnectionError:
            yield "❌ Ollama is not running. Please start it using: `ollama serve`."
        except Exception as e:
            yield f"⚠️ Error: {str(e)}"

import os
from dotenv import load_dotenv
from llama_index.core import Settings

# Prefer Groq embeddings when GROQ_API_KEY is present and groq client is installed.
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load env
base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "../../"))
load_dotenv(os.path.join(project_root, ".env"))


def get_embedding_model(prefer_groq: bool = True):
    """Return an embedding model and set Settings.embed_model.

    If Groq is available and requested, return a thin wrapper that calls
    the Groq embeddings API. Otherwise fall back to HuggingFaceEmbedding.
    """
    if prefer_groq and GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model_name = os.getenv("GROQ_EMBEDDING_MODEL", "bge-base-en-v1.5")

        class GroqEmbeddingWrapper:
            def __init__(self, client, model):
                self.client = client
                self.model = model

            def embed_query(self, text: str):
                # Groq client: client.embeddings.create(model=..., input=...)
                resp = self.client.embeddings.create(model=self.model, input=text)
                return resp.data[0].embedding

            def embed_documents(self, documents):
                # Accept list[str] or list[dict]/Document-like
                inputs = []
                for d in documents:
                    if isinstance(d, str):
                        inputs.append(d)
                    else:
                        # Document objects from llama_index have 'text' attribute
                        inputs.append(getattr(d, "text", str(d)))
                resp = self.client.embeddings.create(model=self.model, input=inputs)
                return [item.embedding for item in resp.data]

        embed_model = GroqEmbeddingWrapper(client, model_name)
        # Do not set Settings.embed_model because LlamaIndex expects a BaseEmbedding instance.
        return embed_model

    # Fallback to local HuggingFace embedding
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    return embed_model
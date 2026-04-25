from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

def get_embedding_model():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    return embed_model
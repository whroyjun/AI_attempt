import chromadb
import time
from chromadb.config import Settings
from chinaai_utils import get_embedding
from emmbedding_query_utils import get_embedding_query

class InMemoryVecDB:
    """
    def __init__(self, name="demo"):
        print("###start to create chromadb")
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.chroma_client.reset()
        self.name = name
        self.collection = self.chroma_client.get_or_create_collection(name=name)
        
    def add_documents(self, documents):
        print("###start to import doc")
        self.collection.add(
        embeddings=[get_embedding(doc) for doc in documents],
        documents=documents,
        metadatas=[{"source": self.name} for _ in documents],
        ids=[f"id_{i}" for i in range(len(documents))]
        )

    def search(self, query, top_n):
        #检索向量数据库
        print("###Start to query vector DB:")
        results = self.collection.query(
            query_embeddings=[get_embedding(query)],
            n_results=top_n
        )

    """    
    def __init__(self, collection_name, embedding_fn,embedding_fn_query):
        print("###start to create chromadb")
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.chroma_client.reset()
        #create an collection
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn=embedding_fn
        self.embedding_fn_query=embedding_fn_query
        
    """

    """ 
    def add_documents(self, documents):
        print("###start to import doc")
        time.sleep(10)
        self.collection.add(
            embeddings=self.embedding_fn(documents),
            #embeddings=[self.embedding_fn(doc) for doc in documents],
            documents=documents,
            ids=[f"id_{i}" for i in range(len(documents))]
       
        )
        print("###importing finished")

    def search(self, query, top_n):
        """检索向量数据库"""
        print("###Start to query vector DB:",query)
        results = self.collection.query(
            query_embeddings=self.embedding_fn_query(query),
            n_results=top_n
        )
        return results
from langchain.document_loaders import TextLoader
from langchain.embeddings import ModelScopeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import chardet
 
# 读取原始文档in
raw_documents_sanguo = TextLoader('/Users/rude3knife/Desktop/三国演义.txt', encoding='utf-16').load()
raw_documents_xiyou = TextLoader('/Users/rude3knife/Desktop/西游记.txt', encoding='utf-16').load()
 
# 分割文档
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents_sanguo = text_splitter.split_documents(raw_documents_sanguo)
documents_xiyou = text_splitter.split_documents(raw_documents_xiyou)
documents = documents_sanguo + documents_xiyou
print("documents nums:", documents.__len__())
 
 
# 生成向量（embedding）
model_id = "damo/nlp_corom_sentence-embedding_chinese-base"
embeddings = ModelScopeEmbeddings(model_id=model_id)
db = Chroma.from_documents(documents, embedding=embeddings)
 
# 检索
query = "美猴王是谁？"
docs = db.similarity_search(query, k=5)
 
# 打印结果
for doc in docs:
    print("===")
    print("metadata:", doc.metadata)
    print("page_content:", doc.page_content)
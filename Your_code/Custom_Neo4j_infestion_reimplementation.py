from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from langflow import CustomComponent
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Neo4jVector
from typing import Dict, Any, List
import logging
from pathlib import Path
import hashlib
import logging
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from typing import List
import re
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
import google.auth
from google.cloud import aiplatform
from langchain_experimental.graph_transformers import LLMGraphTransformer


MODEL_VERSIONS = {
    "openai": "gpt-3.5-turbo",  # Add other model mappings as needed
}

class Neo4jGraphBuilderComponent(CustomComponent):
    
    display_name: str = "Neo4j Graph Builder"
    description: str = "Build a knowledge graph in Neo4j from PDF documents"

    def build_config(self):
        return {
            "path": {"display_name": "PDF Files", "type": "file", "file_types": self.TEXT_FILE_TYPES, "multiple": True},
            "neo4j_uri": {"display_name": "Neo4j URI", "type": "str"},
            "neo4j_username": {"display_name": "Neo4j Username", "type": "str"},
            "neo4j_password": {"display_name": "Neo4j Password", "type": "str", "password": True},
            "openai_api_key": {"display_name": "OpenAI API Key", "type": "str", "password": True},
            "model": {"display_name": "LLM Model", "type": "str", "default": "openai-gpt-3.5"},
            "chunk_size": {"display_name": "Chunk Size", "type": "int", "default": 1000},
            "chunk_overlap": {"display_name": "Chunk Overlap", "type": "int", "default": 200},
            "allowed_nodes": {"display_name": "Allowed Node Types", "type": "str", "default": ""},
            "allowed_relationships": {"display_name": "Allowed Relationship Types", "type": "str", "default": ""},
        }

    TEXT_FILE_TYPES = [".pdf"]  # We're focusing on PDF files for this component

    @staticmethod
    def create_graph_database_connection(uri, userName, password, database):
        enable_user_agent = os.environ.get("ENABLE_USER_AGENT", "False").lower() in ("true", "1", "yes")
        if enable_user_agent:
            graph = Neo4jGraph(url=uri, database=database, username=userName, password=password, refresh_schema=False, sanitize=True,driver_config={'user_agent':os.environ.get('NEO4J_USER_AGENT')})  
        else:
            graph = Neo4jGraph(url=uri, database=database, username=userName, password=password, refresh_schema=False, sanitize=True)    
        return graph
    
    @staticmethod
    def create_edges_between_entities(graph, graph_documents):
        for doc in graph_documents:
            for relationship in doc.relationships:
                source = relationship.source
                target = relationship.target
                rel_type = relationship.type
                
                # Create the edge in Neo4j
                query = (
                    f"MATCH (source:{source.type} {{id: $source_id}}), "
                    f"(target:{target.type} {{id: $target_id}}) "
                    f"MERGE (source)-[r:{rel_type}]->(target)"
                )
                graph.query(query, {
                    "source_id": source.id,
                    "target_id": target.id
                })

        logging.info(f"Created edges between entities")

    @staticmethod
    def check_url_source(source_type, yt_url:str=None, wiki_query:str=None):
        language=''
        try:
            logging.info(f"incoming URL: {yt_url}")
            if source_type == 'youtube':
                if re.match('(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?\/?.*(?:watch|embed)?(?:.*v=|v\/|\/)([\w\-_]+)\&?',yt_url.strip()):
                    youtube_url = Neo4jGraphBuilderComponent.create_youtube_url(yt_url.strip())
                    logging.info(youtube_url)
                    return youtube_url,language
                else:
                    raise Exception('Incoming URL is not youtube URL')
            
            elif  source_type == 'Wikipedia':
                wiki_query_id=''
                wikipedia_url_regex = r'https?:\/\/(www\.)?([a-zA-Z]{2,3})\.wikipedia\.org\/wiki\/(.*)'
                wiki_id_pattern = r'^[a-zA-Z0-9 _\-\.\,\:\(\)\[\]\{\}\/]*$'
                
                match = re.search(wikipedia_url_regex, wiki_query.strip())
                if match:
                        language = match.group(2)
                        wiki_query_id = match.group(3)
                else:
                    raise Exception(f'Not a valid wikipedia url: {wiki_query} ')

                logging.info(f"wikipedia query id = {wiki_query_id}")     
                return wiki_query_id, language     
        except Exception as e:
            logging.error(f"Error in recognize URL: {e}")
            raise Exception(e)

    @staticmethod
    def get_chunk_and_graphDocument(graph_document_list, chunkId_chunkDoc_list):
        logging.info("creating list of chunks and graph documents in get_chunk_and_graphDocument func")
        lst_chunk_chunkId_document=[]
        for graph_document in graph_document_list:            
                for chunk_id in graph_document.source.metadata['combined_chunk_ids'] :
                    lst_chunk_chunkId_document.append({'graph_doc':graph_document,'chunk_id':chunk_id})
                        
        return lst_chunk_chunkId_document  

    @staticmethod
    def load_embedding_model(embedding_model_name: str):
        if embedding_model_name == "openai":
            embeddings = OpenAIEmbeddings()
            dimension = 1536
            logging.info(f"Embedding: Using OpenAI Embeddings , Dimension:{dimension}")
        elif embedding_model_name == "vertexai":        
            embeddings = VertexAIEmbeddings(
                model="textembedding-gecko@003"
            )
            dimension = 768
            logging.info(f"Embedding: Using Vertex AI Embeddings , Dimension:{dimension}")
        else:
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"#, cache_folder="/embedding_model"
            )
            dimension = 384
            logging.info(f"Embedding: Using SentenceTransformer , Dimension:{dimension}")
        return embeddings, dimension

    @staticmethod
    def save_graphDocuments_in_neo4j(graph:Neo4jGraph, graph_document_list:List[GraphDocument]):
        graph.add_graph_documents(graph_document_list)

    @staticmethod
    def handle_backticks_nodes_relationship_id_type(graph_document_list: List[GraphDocument]):
        for graph_document in graph_document_list:
            # Clean node id and types
            cleaned_nodes = []
            for node in graph_document.nodes:
                if node.type.strip() and node.id.strip():
                    node.type = node.type.replace('`', '')
                    cleaned_nodes.append(node)
            
            # Clean relationship id types and source/target node id and types
            cleaned_relationships = []
            for rel in graph_document.relationships:
                if rel.type.strip() and rel.source.id.strip() and rel.source.type.strip() and rel.target.id.strip() and rel.target.type.strip():
                    rel.type = rel.type.replace('`', '')
                    rel.source.type = rel.source.type.replace('`', '')
                    rel.target.type = rel.target.replace('`', '')
                    cleaned_relationships.append(rel)
            
            graph_document.relationships = cleaned_relationships
            graph_document.nodes = cleaned_nodes
        
        return graph_document_list

    @staticmethod
    def delete_uploaded_local_file(merged_file_path, file_name):
        file_path = Path(merged_file_path)
        if file_path.exists():
            file_path.unlink()
            logging.info(f'file {file_name} deleted successfully')

    @staticmethod
    def close_db_connection(graph, api_name):
        if not graph._driver._closed:
            logging.info(f"closing connection for {api_name} api")

    @staticmethod
    def create_gcs_bucket_folder_name_hashed(uri, file_name):
        folder_name = uri + file_name
        folder_name_sha1 = hashlib.sha1(folder_name.encode())
        folder_name_sha1_hashed = folder_name_sha1.hexdigest()
        return folder_name_sha1_hashed

    @staticmethod
    def formatted_time(current_time):
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        return formatted_time

    @staticmethod
    def merge_relationship_between_chunk_and_entites(graph: Neo4jGraph, graph_documents_chunk_chunk_Id : list):
        batch_data = []
        logging.info("Create HAS_ENTITY relationship between chunks and entities")
        chunk_node_id_set = 'id:"{}"'
        for graph_doc_chunk_id in graph_documents_chunk_chunk_Id:
            for node in graph_doc_chunk_id['graph_doc'].nodes:
                query_data={
                    'chunk_id': graph_doc_chunk_id['chunk_id'],
                    'node_type': node.type,
                    'node_id': node.id
                }
                batch_data.append(query_data)
            
        if batch_data:
            unwind_query = """
                        UNWIND $batch_data AS data
                        MATCH (c:Chunk {id: data.chunk_id})
                        CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n
                        MERGE (c)-[:HAS_ENTITY]->(n)
                    """
            graph.query(unwind_query, params={"batch_data": batch_data})

    @staticmethod    
    def update_embedding_create_vector_index(graph, chunkId_chunkDoc_list, file_name):
        #create embedding
        isEmbedding = os.getenv('IS_EMBEDDING', 'FALSE')  # Default to 'FALSE' if not set
        embedding_model = os.getenv('EMBEDDING_MODEL', 'openai')  # Default to 'openai' if not set
        
        embeddings, dimension = Neo4jGraphBuilderComponent.load_embedding_model(embedding_model)
        logging.info(f'embedding model:{embeddings} and dimension:{dimension}')
        data_for_query = []
        logging.info(f"update embedding and vector index for chunks")
        for row in chunkId_chunkDoc_list:
            if isEmbedding.upper() == "TRUE":
                embeddings_arr = embeddings.embed_query(row['chunk_doc'].page_content)
                                        
                data_for_query.append({
                    "chunkId": row['chunk_id'],
                    "embeddings": embeddings_arr
                })
                result = graph.query("SHOW INDEXES YIELD * WHERE labelsOrTypes = ['__Chunk__'] and name = 'vector'")
                vector_index = graph.query("SHOW INDEXES YIELD * WHERE labelsOrTypes = ['Chunk'] and type = 'VECTOR' AND name = 'vector' return options")
                if result:
                    logging.info(f"vector index dropped for 'Chunk'")
                    graph.query("DROP INDEX vector IF EXISTS;")

                if len(vector_index) == 0:
                    logging.info(f'vector index does not exist, will create in next query')
                    graph.query("""CREATE VECTOR INDEX `vector` if not exists for (c:Chunk) on (c.embedding)
                                    OPTIONS {indexConfig: {
                                    `vector.dimensions`: $dimensions,
                                    `vector.similarity_function`: 'cosine'
                                    }}
                                """,
                                {
                                    "dimensions" : dimension
                                }
                                )
        
        if data_for_query:  # Only execute if there's data to update
            query_to_create_embedding = """
                UNWIND $data AS row
                MATCH (d:Document {fileName: $fileName})
                MERGE (c:Chunk {id: row.chunkId})
                SET c.embedding = row.embeddings
                MERGE (c)-[:PART_OF]->(d)
            """       
            graph.query(query_to_create_embedding, params={"fileName":file_name, "data":data_for_query})
        else:
            logging.info("No embeddings to update")

    @staticmethod    
    def create_relation_between_chunks(graph, file_name, chunks: List[Document])->list:
        logging.info("creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks")
        current_chunk_id = ""
        lst_chunks_including_hash = []
        batch_data = []
        relationships = []
        offset=0
        for i, chunk in enumerate(chunks):
            page_content_sha1 = hashlib.sha1(chunk.page_content.encode())
            previous_chunk_id = current_chunk_id
            current_chunk_id = page_content_sha1.hexdigest()
            position = i + 1 
            if i>0:
                offset += len(chunks[i-1].page_content)
            if i == 0:
                firstChunk = True
            else:
                firstChunk = False  
            metadata = {"position": position,"length": len(chunk.page_content), "content_offset":offset}
            chunk_document = Document(
                page_content=chunk.page_content, metadata=metadata
            )
            
            chunk_data = {
                "id": current_chunk_id,
                "pg_content": chunk_document.page_content,
                "position": position,
                "length": chunk_document.metadata["length"],
                "f_name": file_name,
                "previous_id" : previous_chunk_id,
                "content_offset" : offset
            }
            
            if 'page_number' in chunk.metadata:
                chunk_data['page_number'] = chunk.metadata['page_number']
            
            if 'start_timestamp' in chunk.metadata and 'end_timestamp' in chunk.metadata:
                chunk_data['start_time'] = chunk.metadata['start_timestamp']
                chunk_data['end_time'] = chunk.metadata['end_timestamp'] 
                
            batch_data.append(chunk_data)
            
            lst_chunks_including_hash.append({'chunk_id': current_chunk_id, 'chunk_doc': chunk})
            
            # create relationships between chunks
            if firstChunk:
                relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
            else:
                relationships.append({
                    "type": "NEXT_CHUNK",
                    "previous_chunk_id": previous_chunk_id,  # ID of previous chunk
                    "current_chunk_id": current_chunk_id
                })
            
        query_to_create_chunk_and_PART_OF_relation = """
            UNWIND $batch_data AS data
            MERGE (c:Chunk {id: data.id})
            SET c.text = data.pg_content, c.position = data.position, c.length = data.length, c.fileName=data.f_name, c.content_offset=data.content_offset
            WITH data, c
            SET c.page_number = CASE WHEN data.page_number IS NOT NULL THEN data.page_number END,
                c.start_time = CASE WHEN data.start_time IS NOT NULL THEN data.start_time END,
                c.end_time = CASE WHEN data.end_time IS NOT NULL THEN data.end_time END
            WITH data, c
            MATCH (d:Document {fileName: data.f_name})
            MERGE (c)-[:PART_OF]->(d)
        """
        graph.query(query_to_create_chunk_and_PART_OF_relation, params={"batch_data": batch_data})
        
        query_to_create_FIRST_relation = """ 
            UNWIND $relationships AS relationship
            MATCH (d:Document {fileName: $f_name})
            MATCH (c:Chunk {id: relationship.chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                    MERGE (d)-[:FIRST_CHUNK]->(c))
            """
        graph.query(query_to_create_FIRST_relation, params={"f_name": file_name, "relationships": relationships})   
        
        query_to_create_NEXT_CHUNK_relation = """ 
            UNWIND $relationships AS relationship
            MATCH (c:Chunk {id: relationship.current_chunk_id})
            WITH c, relationship
            MATCH (pc:Chunk {id: relationship.previous_chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                    MERGE (c)<-[:NEXT_CHUNK]-(pc))
            """
        graph.query(query_to_create_NEXT_CHUNK_relation, params={"relationships": relationships})   
        
        return lst_chunks_including_hash

    @staticmethod
    def get_llm(model: str, openai_api_key: str):
        """Retrieve the specified language model based on the model name."""
        logging.info(f"Model: {model}")

        model_parts = model.split(',')
        if len(model_parts) == 1:
            # Only model name provided
            model_name = model_parts[0]
            llm = ChatOpenAI(
                api_key=openai_api_key,
                model=model_name,
                temperature=0,
            )
        elif len(model_parts) == 3:
            # Model name, API endpoint, and API key provided
            model_name, api_endpoint, api_key = model_parts
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=api_endpoint,
                model=model_name,
                temperature=0,
            )
        else:
            raise ValueError("Invalid model format. Expected either 'model_name' or 'model_name,api_endpoint,api_key'")

        logging.info(f"Model created - Model Version: {model_name}")
        return llm, model_name

    @staticmethod
    def get_combined_chunks(chunkId_chunkDoc_list):
        # Use a default value of 1 if the environment variable is not set
        chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE", "1"))
        logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
        combined_chunk_document_list = []
        combined_chunks_page_content = [
            "".join(
                document["chunk_doc"].page_content
                for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
            )
            for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
        ]
        combined_chunks_ids = [
            [
                document["chunk_id"]
                for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
            ]
            for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
        ]

        for i in range(len(combined_chunks_page_content)):
            combined_chunk_document_list.append(
                Document(
                    page_content=combined_chunks_page_content[i],
                    metadata={"combined_chunk_ids": combined_chunks_ids[i]},
                )
            )
        return combined_chunk_document_list

    @staticmethod
    def get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship
    ):
        futures = []
        graph_document_list = []

        if "diffbot_api_key" in dir(llm):
            llm_transformer = llm
        else:
            if "get_name" in dir(llm) and llm.get_name() == "ChatOllama":
                node_properties = False
            else:
                node_properties = ["description"]
            llm_transformer = LLMGraphTransformer(
                llm=llm,
                node_properties=node_properties,
                allowed_nodes=allowedNodes,
                allowed_relationships=allowedRelationship,
            )
        with ThreadPoolExecutor(max_workers=10) as executor:
            for chunk in combined_chunk_document_list:
                chunk_doc = Document(
                    page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
                )
                futures.append(
                    executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc])
                )

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                graph_document = future.result()
                graph_document_list.append(graph_document[0])

        return graph_document_list

    @staticmethod
    def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship, openai_api_key):
        
        llm, model_name = Neo4jGraphBuilderComponent.get_llm(model, openai_api_key)
        combined_chunk_document_list = Neo4jGraphBuilderComponent.get_combined_chunks(chunkId_chunkDoc_list)
        
        if  allowedNodes is None or allowedNodes=="":
            allowedNodes =[]
        else:
            allowedNodes = allowedNodes.split(',')    
        if  allowedRelationship is None or allowedRelationship=="":   
            allowedRelationship=[]
        else:
            allowedRelationship = allowedRelationship.split(',')
            
        graph_document_list = Neo4jGraphBuilderComponent.get_graph_document_list(
            llm, combined_chunk_document_list, allowedNodes, allowedRelationship
        )
        return graph_document_list


    def build(self, 
              path: List[str],
              neo4j_uri: str,
              neo4j_username: str,
              neo4j_password: str,
              openai_api_key: str,
              model: str = "gpt-3.5-turbo",
              chunk_size: int = 1000,
              chunk_overlap: int = 200,
              allowed_nodes: str = "",
              allowed_relationships: str = "") -> Dict[str, Any]:
        
        logging.info("Starting graph building process")
        
        graph = self.create_graph_database_connection(neo4j_uri, neo4j_username, neo4j_password, "neo4j")
        
        all_chunks = []
        
        for pdf_path in path:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
        
        # Create chunk nodes and relationships
        chunkId_chunkDoc_list = self.create_relation_between_chunks(graph, Path(pdf_path).name, all_chunks)

        # Update embeddings and create vector index
        self.update_embedding_create_vector_index(graph, chunkId_chunkDoc_list, Path(pdf_path).name)

        # Get graph documents from LLM
        graph_documents = self.get_graph_from_llm(model, chunkId_chunkDoc_list, allowed_nodes, allowed_relationships, openai_api_key)

        # Clean and save graph documents
        cleaned_graph_documents = self.handle_backticks_nodes_relationship_id_type(graph_documents)
        self.save_graphDocuments_in_neo4j(graph, cleaned_graph_documents)

        # Create relationships between chunks and entities
        chunks_and_graphDocuments_list = self.get_chunk_and_graphDocument(cleaned_graph_documents, chunkId_chunkDoc_list)
        self.merge_relationship_between_chunk_and_entites(graph, chunks_and_graphDocuments_list)
        
        # Initialize embeddings for vector search
        embeddings, _ = self.load_embedding_model("openai")
        
        # Initialize Neo4j Vector Store
        vector_store = Neo4jVector.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            index_name="document_index",
            node_label="Document",
            text_node_property="text",
            embedding_node_property="embedding",
            create_id_index=True,
        )
        
        logging.info(f"Built graph with {len(all_chunks)} chunks from {len(path)} PDF files")
        
        # Close the database connection
        self.close_db_connection(graph, "build")
        
        return {
            "output": f"Successfully built graph with {len(all_chunks)} chunks from {len(path)} PDF files",
            "num_chunks": len(all_chunks),
            "num_files": len(path)
        }

    @staticmethod
    def resolve_path(file_path: str) -> str:
        return str(Path(file_path).resolve())
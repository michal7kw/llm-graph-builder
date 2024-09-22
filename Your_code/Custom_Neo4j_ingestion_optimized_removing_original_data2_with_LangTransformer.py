from langflow.custom import Component
from langflow.io import HandleInput, Output, DropdownInput, StrInput, SecretStrInput
from langflow.schema import Data
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import logging
from datetime import datetime
from langchain.docstore.document import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import os
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_document import Node


class Neo4jIngestionComponent(Component):
    display_name = "Custom_Neo4j_ingestion_optimized"
    description = "Load pre-chunked data into Neo4j using LLMGraphTransformer"
    icon = "git-graph"
    name = "NEO4J"

    inputs = [
        HandleInput(
            name="splits",
            display_name="Pre-chunked Data",
            input_types=["Data"],
            is_list=True,
        ),
        DropdownInput(
            name="openai_embedding_model",
            display_name="Embedding Model",
            advanced=False,
            options=[
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            value="text-embedding-3-small",
        ),
        StrInput(name="openai_endpoint", display_name="OpenAI Embeddings Endpoint"),
        SecretStrInput(name="openai_api_key", display_name="OpenAI API Key", value="OPENAI_API_KEY"),
        StrInput(name="neo4j_uri", display_name="NEO4J URI"),
        StrInput(name="neo4j_username", display_name="NEO4J USERNAME"),
        StrInput(name="neo4j_password", display_name="NEO4J PASSWORD"),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_logger()
        self.kg = None

    def setup_logger(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def process_chunks(self, splits):
        self.logger.info("Processing pre-chunked data")
        documents = {}
        chunks = []

        for split in splits:
            chunk_data = split.data
            chunks.append({
                "chunkId": chunk_data['id'],
                "text": chunk_data['chunk'],
                "source": chunk_data['source'],
                "pageNumber": chunk_data['page'],
                "fileName": chunk_data['file_name'],
            })
            
            if chunk_data['file_name'] not in documents:
                documents[chunk_data['file_name']] = {
                    "fileName": chunk_data['file_name'],
                    "source": chunk_data['source']
                }
                
        self.logger.info(f"Processed {len(chunks)} chunks from {len(documents)} documents")
        return chunks, documents

    def clear_database(self):
        self.logger.info("Clearing database")
        self.kg.query("MATCH (n) DETACH DELETE n")
        self.kg.query("DROP CONSTRAINT unique_chunk IF EXISTS")
        self.kg.query("DROP CONSTRAINT unique_document IF EXISTS")
        self.kg.query("DROP INDEX chunks_embedding IF EXISTS")

    def create_document_nodes(self, documents):
        self.logger.info("Creating document nodes")
        merge_document_node = """
        MERGE (d:Document {fileName: $params.fileName})
        ON CREATE SET
            d.source = $params.source
        RETURN d
        """
        node_count = 0
        for doc in documents.values():
            self.kg.query(merge_document_node, params={'params': doc})
            node_count += 1
        
        self.logger.info(f"Created {node_count} document nodes")
        return node_count

    def create_chunk_nodes(self, chunks):
        self.logger.info("Creating chunk nodes")
        merge_chunk_node = """
        MERGE (c:Chunk {chunkId: $params.chunkId})
        ON CREATE SET 
            c.text = $params.text,
            c.source = $params.source,
            c.pageNumber = $params.pageNumber,
            c.fileName = $params.fileName
        RETURN c
        """
        
        node_count = 0
        for chunk in chunks:
            self.logger.debug(f"Creating `:Chunk` node for chunk ID {chunk['chunkId']}")
            self.kg.query(merge_chunk_node, params={'params': chunk})
            node_count += 1
            
        self.logger.info(f"Created {node_count} chunk nodes")
        return node_count

    def create_constraints_and_indexes(self):
        self.logger.info("Creating constraints and indexes")
        self.kg.query("CREATE CONSTRAINT unique_chunk IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE")
        self.kg.query("CREATE CONSTRAINT unique_document IF NOT EXISTS FOR (d:Document) REQUIRE d.fileName IS UNIQUE")
        self.kg.query("""
         CREATE VECTOR INDEX chunks_embedding IF NOT EXISTS
          FOR (c:Chunk) ON (c.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
         }}""")

    def create_embeddings(self):
        self.logger.info("Creating embeddings")
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=self.openai_embedding_model)
        self.kg.query("""
        MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
        WITH chunk
        CALL apoc.when(chunk.text IS NOT NULL,
            'RETURN genai.vector.encode($text, "OpenAI", $config) AS vector',
            'RETURN [] AS vector',
            {text: chunk.text, config: $config}
        ) YIELD value
        CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", value.vector)
        RETURN count(chunk)
        """, params={
            "config": {
                "token": self.openai_api_key,
                "endpoint": self.openai_endpoint,
                "model": self.openai_embedding_model
            }
        })

    def link_chunks_to_documents(self):
        self.logger.info("Linking chunks to documents")
        nodes_belongs_to_query = """
        MATCH (c:Chunk), (d:Document)
        WHERE c.fileName = d.fileName
        MERGE (c)-[:BELONGS_TO]->(d)
        """
        self.kg.query(nodes_belongs_to_query)

    def get_llm(self):
        self.logger.info("Initializing LLM")
        return ChatOpenAI(api_key=self.openai_api_key, model="gpt-3.5-turbo", temperature=0)

    def get_combined_chunks(self, chunkId_chunkDoc_list):
        chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE", "1"))
        self.logger.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
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

    def get_graph_document_list(self, llm, combined_chunk_document_list, allowedNodes, allowedRelationship):
        graph_document_list = []

        node_properties = ["description"]
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            node_properties=node_properties,
            allowed_nodes=allowedNodes,
            allowed_relationships=allowedRelationship,
        )
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for chunk in combined_chunk_document_list:
                chunk_doc = Document(
                    page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
                )
                futures.append(
                    executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc])
                )
            
            for future in executor.map(lambda f: f.result(), futures):
                graph_document_list.extend(future)

        return graph_document_list

    def get_graph_from_llm(self, chunks):
        self.logger.info("Getting graph from LLM")
        
        llm = self.get_llm()
        chunkId_chunkDoc_list = [{"chunk_id": chunk["chunkId"], "chunk_doc": Document(page_content=chunk["text"], metadata={"id": chunk["chunkId"]})} for chunk in chunks]
        combined_chunk_document_list = self.get_combined_chunks(chunkId_chunkDoc_list)
        
        allowedNodes = []
        allowedRelationship = []
        
        graph_document_list = self.get_graph_document_list(
            llm, combined_chunk_document_list, allowedNodes, allowedRelationship
        )
        return graph_document_list

    def build_output(self) -> Data:
        self.logger.info("Starting build_output method")
        self.kg = Neo4jGraph(
            url=self.neo4j_uri, 
            username=self.neo4j_username, 
            password=self.neo4j_password
        )
        
        chunks, documents = self.process_chunks(self.splits)
        
        self.clear_database()
        document_count = self.create_document_nodes(documents)
        chunk_count = self.create_chunk_nodes(chunks)
        self.create_constraints_and_indexes()
        self.create_embeddings()
        self.link_chunks_to_documents()
        
        graph_documents = self.get_graph_from_llm(chunks)
        self.save_graph_documents(graph_documents)
        self.merge_relationships_between_chunk_and_entities(graph_documents)
        
        total_nodes = document_count + chunk_count
        entity_count = sum(len(doc.nodes) for doc in graph_documents)
        relationship_count = sum(len(doc.relationships) for doc in graph_documents)
        
        return Data(text=f"Created {total_nodes} nodes ({document_count} documents, {chunk_count} chunks), {entity_count} entities, and {relationship_count} relationships")

    def save_graph_documents(self, graph_documents):
        self.logger.info("Saving graph documents to Neo4j")
        self.kg.add_graph_documents(graph_documents)

    def merge_relationships_between_chunk_and_entities(self, graph_documents):
        self.logger.info("Creating HAS_ENTITY relationships between chunks and entities")
        batch_data = []
        for doc in graph_documents:
            for node in doc.nodes:
                batch_data.append({
                    'chunk_id': doc.source.metadata['id'],
                    'node_type': node.type,
                    'node_id': node.id
                })

        if batch_data:
            unwind_query = """
                UNWIND $batch_data AS data
                MATCH (c:Chunk {chunkId: data.chunk_id})
                CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n
                MERGE (c)-[:HAS_ENTITY]->(n)
            """
            self.kg.query(unwind_query, params={"batch_data": batch_data})
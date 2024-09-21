from langflow.custom import Component
from langflow.io import HandleInput, Output, DropdownInput, StrInput, SecretStrInput
from langflow.schema import Data
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
import logging
from datetime import datetime

class Neo4jIngestionComponent(Component):
    display_name = "Custom_Neo4j_ingestion_optimized"
    description = "Load data into Neo4j"
    icon = "git-graph"
    name = "NEO4J"

    inputs = [
        HandleInput(
            name="splits",
            display_name="PDF Reader Output",
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
        self.kg = None  # Will be initialized in build_output

    def setup_logger(self):
        """Set up a logger for this component."""
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def build_chunks(self, splits):
        """
        Build chunks and documents from the input splits.
        
        :param splits: List of split objects from PDF reader
        :return: Tuple of (chunks, documents)
        """
        self.logger.info("Building chunks and documents")
        documents = {}
        chunks = []

        for chunk_seq_id, split in enumerate(splits, start=1):
            chunks.append({
                "chunkId": split.id,
                "text": split.chunk,
                "source": split.source,
                "pageNumber": split.page,
                "fileName": split.file_name,
                "chunkSeqId": chunk_seq_id
            })
            
            if split.file_name not in documents:
                documents[split.file_name] = {
                    "fileName": split.file_name,
                    "source": split.source
                }
                
        self.logger.info(f"Built {len(chunks)} chunks and {len(documents)} documents")
        return chunks, documents

    def clear_database(self):
        """Clear existing data and indexes from the database."""
        self.logger.info("Clearing database")
        self.kg.query("MATCH (n) DETACH DELETE n")
        self.kg.query("DROP CONSTRAINT unique_chunk IF EXISTS")
        self.kg.query("DROP CONSTRAINT unique_document IF EXISTS")
        self.kg.query("DROP INDEX chunks_embedding IF EXISTS")

    def create_document_nodes(self, documents):
        """
        Create document nodes in the graph.
        
        :param documents: Dictionary of document information
        :return: Number of created nodes
        """
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
        """
        Create chunk nodes in the graph.
        
        :param chunks: List of chunk dictionaries
        :return: Number of created nodes
        """
        self.logger.info("Creating chunk nodes")
        merge_chunk_node = """
        MERGE (c:Chunk {chunkId: $params.chunkId})
        ON CREATE SET 
            c.text = $params.text,
            c.source = $params.source,
            c.pageNumber = $params.pageNumber,
            c.fileName = $params.fileName,
            c.chunkSeqId = $params.chunkSeqId
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
        """Create necessary constraints and indexes in the graph."""
        self.logger.info("Creating constraints and indexes")
        self.kg.query("""
        CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
        FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
        """)
        
        self.kg.query("""
        CREATE CONSTRAINT unique_document IF NOT EXISTS 
        FOR (d:Document) REQUIRE d.fileName IS UNIQUE
        """)
        
        self.kg.query("""
         CREATE VECTOR INDEX chunks_embedding IF NOT EXISTS
          FOR (c:Chunk) ON (c.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
         }}""")

    def create_embeddings(self):
        """Create embeddings for chunk nodes."""
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

    def link_chunks_sequentially(self):
        """Link chunk nodes sequentially within each document."""
        self.logger.info("Linking chunks sequentially")
        nodes_next_query = """
        MATCH (chunk:Chunk)
        WITH chunk.fileName AS fileName, collect(chunk) AS chunks
        UNWIND range(0, size(chunks)-2) AS i
        WITH chunks[i] AS current, chunks[i+1] AS next
        WHERE current.fileName = next.fileName
        MERGE (current)-[:NEXT]->(next)
        """
        self.kg.query(nodes_next_query)

    def link_chunks_to_documents(self):
        """Link chunk nodes to their respective document nodes."""
        self.logger.info("Linking chunks to documents")
        nodes_belongs_to_query = """
        MATCH (c:Chunk), (d:Document)
        WHERE c.fileName = d.fileName
        MERGE (c)-[:BELONGS_TO]->(d)
        """
        self.kg.query(nodes_belongs_to_query)

    def create_index_if_not_exists(self, query, index_description):
        """
        Create an index if it doesn't already exist.
        
        :param query: The query to create the index
        :param index_description: A description of the index for logging
        """
        try:
            self.kg.query(query)
            self.logger.info(f"Created index: {index_description}")
        except Exception as e:
            if "ConstraintAlreadyExists" in str(e) or "IndexAlreadyExists" in str(e):
                self.logger.info(f"Index or constraint already exists: {index_description}. Skipping creation.")
            else:
                self.logger.error(f"Error creating index {index_description}: {str(e)}")

    def create_additional_indexes(self):
        """Create additional indexes for optimized querying."""
        self.logger.info("Creating additional indexes")
        self.create_index_if_not_exists(
            "CREATE INDEX FOR (c:Chunk) ON (c.fileName) IF NOT EXISTS",
            "Chunk.fileName"
        )
        self.create_index_if_not_exists(
            "CREATE INDEX FOR (c:Chunk) ON (c.pageNumber) IF NOT EXISTS",
            "Chunk.pageNumber"
        )
        self.create_index_if_not_exists(
            "CREATE INDEX FOR (d:Document) ON (d.fileName) IF NOT EXISTS",
            "Document.fileName"
        )

    def build_output(self) -> Data:
        """
        Main method to build the Neo4j graph from the input data.
        
        :return: Data object with information about created nodes
        """
        self.logger.info("Starting build_output method")
        self.kg = Neo4jGraph(
            url=self.neo4j_uri, 
            username=self.neo4j_username, 
            password=self.neo4j_password
        )
        
        chunks, documents = self.build_chunks(self.splits)
        
        self.clear_database()
        document_count = self.create_document_nodes(documents)
        chunk_count = self.create_chunk_nodes(chunks)
        self.create_constraints_and_indexes()
        self.create_embeddings()
        self.link_chunks_sequentially()
        self.link_chunks_to_documents()
        self.create_additional_indexes()
        
        total_nodes = document_count + chunk_count
        return Data(text=f"Created {total_nodes} nodes ({document_count} documents, {chunk_count} chunks)")
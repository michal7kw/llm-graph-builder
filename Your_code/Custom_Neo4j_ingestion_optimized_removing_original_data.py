from langflow.custom import Component
from langflow.io import HandleInput, Output
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
        Build chunks and parents from the input splits.
        
        :param splits: List of split objects from PDF reader
        :return: Tuple of (chunks, parents)
        """
        self.logger.info("Building chunks and parents")
        parents = {}
        chunks = []

        for chunk_seq_id, split in enumerate(splits, start=1):
            chunks.append({
                "chunkId": split.id,
                "text": split.chunk,
                "source": split.file_name,
                "pageId": split.page,
                "chunkSeqId": chunk_seq_id
            })
            
            if split.page not in parents:
                parents[split.page] = {"text": split.chunk, "pageId": split.page}
                
        self.logger.info(f"Built {len(chunks)} chunks and {len(parents)} parents")
        return chunks, parents

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
            if "EquivalentSchemaRuleAlreadyExists" in str(e):
                self.logger.info(f"Index already exists: {index_description}. Skipping creation.")
            else:
                self.logger.error(f"Error creating index {index_description}: {str(e)}")
                raise e

    def clear_database(self):
        """Clear existing data and indexes from the database."""
        self.logger.info("Clearing database")
        self.kg.query("MATCH (n) DETACH DELETE n")
        self.kg.query("DROP CONSTRAINT unique_chunk IF EXISTS")
        self.kg.query("DROP INDEX unique_chunk IF EXISTS")
        self.kg.query("DROP INDEX chunks_embedding IF EXISTS")

    def create_parent_nodes(self, parents):
        """
        Create parent nodes in the graph.
        
        :param parents: Dictionary of parent nodes
        :return: Number of created nodes
        """
        self.logger.info("Creating parent nodes")
        merge_parent_node = """
        MERGE(p:Page {pageId: $params.pageId})
        ON CREATE SET
            p.text = $params.text
        RETURN p
        """
        node_count = 0
        for p in parents:
            self.kg.query(merge_parent_node, params={'params': parents[p]})
            node_count += 1
        
        self.logger.info(f"Created {node_count} parent nodes")
        return node_count

    def create_chunk_nodes(self, chunks):
        """
        Create chunk nodes in the graph.
        
        :param chunks: List of chunk dictionaries
        :return: Number of created nodes
        """
        self.logger.info("Creating chunk nodes")
        merge_chunk_node = """
        MERGE(c:Chunk {chunkId: $params.chunkId})
        ON CREATE SET 
            c.text = $params.text,
            c.source = $params.source,
            c.pageId = $params.pageId,
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
         CREATE VECTOR INDEX `chunks_embedding` IF NOT EXISTS
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
        """Link chunk nodes sequentially."""
        self.logger.info("Linking chunks sequentially")
        nodes_next_query = """
          MATCH (chunk:Chunk)
          WITH chunk
            ORDER BY chunk.chunkSeqId ASC
          WITH collect(chunk) as chunk_list
            CALL apoc.nodes.link(
                chunk_list, 
                "NEXT", 
                {avoidDuplicates: true}
            )
          RETURN size(chunk_list)
        """
        self.kg.query(nodes_next_query)

    def link_chunks_to_pages(self):
        """Link chunk nodes to their respective page nodes."""
        self.logger.info("Linking chunks to pages")
        nodes_belongs_to_query = """
        MATCH (c:Chunk), (p:Page)
        WHERE c.pageId = p.pageId
        MERGE (c)-[:BELONGS_TO]->(p)
        """
        self.kg.query(nodes_belongs_to_query)

    def create_additional_indexes(self):
        """Create additional indexes for optimized querying."""
        self.logger.info("Creating additional indexes")
        self.create_index_if_not_exists("CREATE INDEX FOR (c:Chunk) ON (c.pageId)", "Chunk.pageId")
        self.create_index_if_not_exists("CREATE INDEX FOR (p:Page) ON (p.pageId)", "Page.pageId")

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
        
        chunks, parents = self.build_chunks(self.splits)
        
        self.clear_database()
        parent_count = self.create_parent_nodes(parents)
        chunk_count = self.create_chunk_nodes(chunks)
        self.create_constraints_and_indexes()
        self.create_embeddings()
        self.link_chunks_sequentially()
        self.link_chunks_to_pages()
        self.create_additional_indexes()
        
        total_nodes = parent_count + chunk_count
        return Data(text=f"Created {total_nodes} nodes ({parent_count} parents, {chunk_count} chunks)")
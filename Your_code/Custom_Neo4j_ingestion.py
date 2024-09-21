from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema import Data
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
import logging
from datetime import datetime

class Neo4jIngestionComponent(Component):
    display_name = "NEO4J"
    description = "Custom_Neo4j_ingestion"
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

    def setup_logger(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def build_chunks(self, splits):
        self.logger.info("Building chunks and parents")
        parents = {}
        chunks = []
        chunk_seq_id = 0

        for split in splits:
            chunk_seq_id += 1
            chunks.append({
                "chunkId": split.id,
                "text": split.chunk,
                "source": split.source,
                "pageId": split.page,
                "chunkSeqId": chunk_seq_id
            })
            
            if split.page not in parents:
                parents[split.page] = {"text": split.text, "pageId": split.page}
                
        self.logger.info(f"Built {len(chunks)} chunks and {len(parents)} parents")
        return chunks, parents

    def build_output(self) -> Data:
        self.logger.info("Starting build_output method")
        kg = Neo4jGraph(
            url=self.neo4j_uri, 
            username=self.neo4j_username, 
            password=self.neo4j_password
        )
        
        chunks, parents = self.build_chunks(self.splits)
        
        self.logger.info("Clearing database")
        kg.query("MATCH (n) DETACH DELETE n")
        kg.query("DROP CONSTRAINT unique_chunk IF EXISTS")
        kg.query("DROP INDEX unique_chunk IF EXISTS")
        kg.query("DROP INDEX chunks_embedding IF EXISTS")
        
        self.logger.info("Creating parent nodes")
        merge_parent_node = """
        MERGE(p:Page {pageId: $params.pageId})
        ON CREATE SET
            p.text = $params.text
        RETURN p
        """
        node_count = 0
        for p in parents:
            kg.query(merge_parent_node, params={'params': parents[p]})
            node_count += 1
        
        self.logger.info(f"Created {node_count} parent nodes")
        
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
            kg.query(merge_chunk_node, params={'params': chunk})
            node_count += 1
            
        self.logger.info(f"Created {node_count} chunk nodes")
        
        self.logger.info("Creating constraints and indexes")
        kg.query("""
        CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
        FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
        """)
        
        kg.query("""
         CREATE VECTOR INDEX `chunks_embedding` IF NOT EXISTS
          FOR (c:Chunk) ON (c.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
         }}""")

        self.logger.info("Creating embeddings")
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=self.openai_embedding_model)
        kg.query("""
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
        
        kg.query(nodes_next_query)
        
        self.logger.info("Linking chunks to pages")
        nodes_belongs_to_query = """
        MATCH (c:Chunk), (p:Page)
        WHERE c.pageId = p.pageId
        MERGE (c)-[:BELONGS_TO]->(p)
        """
        
        kg.query(nodes_belongs_to_query)
        
        self.logger.info("Creating additional indexes")
        try:
            kg.query("CREATE INDEX FOR (c:Chunk) ON (c.pageId)")
        except Exception as e:
            if "EquivalentSchemaRuleAlreadyExists" in str(e):
                self.logger.info("Index for (c:Chunk) ON (c.pageId) already exists. Skipping creation.")
            else:
                raise e

        try:
            kg.query("CREATE INDEX FOR (p:Page) ON (p.pageId)")
        except Exception as e:
            if "EquivalentSchemaRuleAlreadyExists" in str(e):
                self.logger.info("Index for (p:Page) ON (p.pageId) already exists. Skipping creation.")
            else:
                raise e
        
        return Data(text=f"Created {node_count} nodes")
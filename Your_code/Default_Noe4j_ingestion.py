from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema import Data
from langchain_community.graphs import Neo4jGraph

class Neo4jIngestionComponent(Component):
    display_name = "NEO4J"
    description = "Default_Noe4j_ingestion"
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
            name="openai_model",
            display_name="Model",
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

    def build_chunks(self, splits):
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
                
        return chunks, parents
        
    def build_output(self) -> Data:
        kg = Neo4jGraph(
            url=self.neo4j_uri, 
            username=self.neo4j_username, 
            password=self.neo4j_password
        )
        
        chunks, parents = self.build_chunks(self.splits)
        
        ## CLEAR DATABASE
    
        kg.query(""" MATCH (n) DETACH DELETE n """)
        kg.query(""" DROP CONSTRAINT unique_chunk IF EXISTS """)
        kg.query(""" DROP INDEX unique_chunk IF EXISTS """)
        kg.query(""" DROP INDEX chunks_embedding IF EXISTS """)
        
    
        ## CREATE PARENT NODES 
        
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
        
        print(f"Created {node_count} parent nodes")
        
        ## CREATE NODE WITH PROPERTIES

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
            print(f"Creating `:Chunk` node for chunk ID {chunk['chunkId']}")
            kg.query(merge_chunk_node, params={'params': chunk})
            node_count += 1
            
        print(f"Created {node_count} chunk nodes")
        
        ## CREATE UNIQUE INDEX 
        
        kg.query("""
        CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
        FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
        """)
        
        ## CREATE VECTOR INDEX

        kg.query("""
         CREATE VECTOR INDEX `chunks_embedding` IF NOT EXISTS
          FOR (c:Chunk) ON (c.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
         }}""")

        ## DISPLAY INDEXES 
        # print(kg.query("SHOW INDEXES"))
        
        kg.query("""
        MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
        WITH chunk, genai.vector.encode(
            chunk.text, 
            "OpenAI", 
              {
                token: $openAiApiKey, 
                endpoint: $openAiEndpoint,
                model: $openAiModel
              }) AS vector
            CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
        """, params={
                "openAiApiKey":self.openai_api_key, 
                "openAiEndpoint": self.openai_endpoint,
                "openAiModel": self.openai_model,
        })
        
        
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
        
        nodes_belongs_to_query = """
        MATCH (c:Chunk), (p:Page)
        WHERE c.pageId = p.pageId
        MERGE (c)-[:BELONGS_TO]->(p)
        """
        
        kg.query(nodes_belongs_to_query)
        
        index_query = """
        CREATE INDEX FOR (c:Chunk) ON (c.pageId);
        CREATE INDEX FOR (p:Page) ON (p.pageId);
        """
        
        kg.query(index_query)
        
        return Data(text=f"Created {node_count} nodes")

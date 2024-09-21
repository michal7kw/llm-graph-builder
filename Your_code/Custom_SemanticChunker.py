from typing import List
import uuid

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_text_splitters import CharacterTextSplitter

from langflow.custom import Component
from langflow.io import HandleInput, IntInput, MessageTextInput, Output
from langflow.schema import Data
from langflow.utils.util import unescape_string


class SplitTextComponent(Component):
    display_name: str = "Split Text"
    description: str = "Split text into chunks based on specified criteria."
    icon = "scissors-line-dashed"
    name = "SplitText"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Data Inputs",
            info="The data to split.",
            input_types=["Data"],
            is_list=True,
        ),
        MessageTextInput(
            name="openai_api_key",
            display_name="OpenAI API Key",
            info="Your OpenAI API key for embeddings.",
        ),
        SelectInput(
            name="breakpoint_threshold_type",
            display_name="Breakpoint Threshold Type",
            info="The method to determine the threshold for splitting.",
            options=["percentile", "standard_deviation", "interquartile", "gradient"],
            value="percentile",
        ),
        # IntInput(
        #     name="chunk_overlap",
        #     display_name="Chunk Overlap",
        #     info="Number of characters to overlap between chunks.",
        #     value=200,
        # ),
        # IntInput(
        #     name="chunk_size",
        #     display_name="Chunk Size",
        #     info="The maximum number of characters in each chunk.",
        #     value=1000,
        # ),
        # MessageTextInput(
        #     name="separator",
        #     display_name="Separator",
        #     info="The character to split on. Defaults to newline.",
        #     value="\n",
        # ),
    ]

    outputs = [
        Output(display_name="Chunks", name="chunks", method="split_text"),
    ]

    def _docs_to_data(self, docs):
        data = []
        for i, doc in enumerate(docs):
            chunk_id = str(uuid.uuid4())
            data.append(Data(
                id=chunk_id,
                chunk=doc.page_content,
                source=doc.metadata.get('source', ''),
                page=doc.metadata.get('page', 0),
                text=doc.metadata.get('text', '')
            ))
        return data

    def split_text(self) -> List[Data]:
        documents = []
        for _input in self.data_inputs:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())

        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type
        )
        
        docs = splitter.create_documents([doc.page_content for doc in documents])
        data = self._docs_to_data(docs)
        self.status = data
        return data

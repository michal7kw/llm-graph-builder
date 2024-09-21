from typing import List, Dict, Any
import uuid

from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document

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
        IntInput(
            name="chunk_overlap",
            display_name="Chunk Overlap",
            info="Number of characters to overlap between chunks.",
            value=200,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            info="The maximum number of characters in each chunk.",
            value=1000,
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            info="The character to split on. Defaults to newline.",
            value="\n",
        ),
    ]

    outputs = [
        Output(display_name="Chunks", name="chunks", method="split_text"),
    ]

    def _split_document(self, doc: Dict[str, Any], splitter: CharacterTextSplitter) -> List[Data]:
        chunks = splitter.split_text(doc['page_content'])
        return [
            Data(
                data={
                    'id': str(uuid.uuid4()),
                    'chunk': chunk,
                    'source': doc['metadata'].get('source', ''),
                    'page': doc['metadata'].get('page_number', 0),
                    'file_name': doc['metadata'].get('filename', ''),
                }
            ) for chunk in chunks
        ]

    def split_text(self) -> List[Data]:
        separator = unescape_string(self.separator)

        splitter = CharacterTextSplitter(
            chunk_overlap=self.chunk_overlap,
            chunk_size=self.chunk_size,
            separator=separator,
        )

        all_chunks = []
        for _input in self.data_inputs:
            if isinstance(_input, Data) and isinstance(_input.data, dict):
                pages = _input.data.get('pages', [])
                if isinstance(pages, list) and all(isinstance(doc, dict) and 'page_content' in doc for doc in pages):
                    for doc in pages:
                        chunks = self._split_document(doc, splitter)
                        all_chunks.extend(chunks)
                else:
                    # If it's not a list of document-like dictionaries, skip it
                    continue

        print("SplitTextComponent output:", all_chunks)
        self.status = all_chunks
        return all_chunks
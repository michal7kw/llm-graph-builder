# copy of the PDFComponent but with the get_documents_from_file_by_path method (fille: local_file.py)
# extract_graph_from_file_local_file() --> get_documents_from_file_by_path()

# this component is to load pdf files and return the pages as a list of documents
# Status: working

from pathlib import Path
from langflow.base.data.utils import parse_text_file_to_data
from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from langchain_community.document_loaders import UnstructuredFileLoader
import logging
from langflow.custom import Component
from langflow.io import BoolInput, FileInput, Output
from langflow.schema import Data
from langchain_core.documents import Document


class PDFComponent(Component):
    display_name = "PDF"
    description = "A PDF file loader."
    icon = "file-text"

    inputs = [
        FileInput(
            name="file_path",
            display_name="Path",
            file_types=["pdf"],
            info=f"Supported file types: pdf",
        ),
        # BoolInput(
        #     name="silent_errors",
        #     display_name="Silent Errors",
        #     advanced=True,
        #     info="If true, errors will not raise an exception.",
        # ),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="get_documents_from_file_by_path"),
    ]

    def get_pages_with_page_numbers(unstructured_pages):
        # Initialize variables to store processed pages and current page information
        pages = []
        page_number = 1
        page_content = ''
        metadata = {}

        for page in unstructured_pages:
            if 'page_number' in page.metadata:
                # If the page has a page number in its metadata
                if page.metadata['page_number'] == page_number:
                    # Accumulate content for the current page
                    page_content += page.page_content
                    # Update metadata for the current page
                    metadata = {
                        'source': page.metadata['source'],
                        'page_number': page_number,
                        'filename': page.metadata['filename'],
                        'filetype': page.metadata['filetype']
                    }
                
                if page.metadata['page_number'] > page_number:
                    # If we've moved to a new page, save the previous page and reset
                    page_number += 1
                    pages.append(Document(page_content=page_content))
                    page_content = ''
                
                if page == unstructured_pages[-1]:
                    # If it's the last page, add it to the list
                    pages.append(Document(page_content=page_content))
                    
            elif page.metadata['category'] == 'PageBreak' and page != unstructured_pages[0]:
                # If we encounter a page break (except for the first page)
                page_number += 1
                pages.append(Document(page_content=page_content, metadata=metadata))
                page_content = ''
                metadata = {}
            else:
                # For pages without explicit page numbers or breaks
                page_content += page.page_content
                metadata_with_custom_page_number = {
                    'source': page.metadata['source'],
                    'page_number': 1,  # Assume it's the first page if no explicit numbering
                    'filename': page.metadata['filename'],
                    'filetype': page.metadata['filetype']
                }
                if page == unstructured_pages[-1]:
                    # If it's the last page, add it to the list
                    pages.append(Document(page_content=page_content, metadata=metadata_with_custom_page_number))
        
        return pages

    def load_document_content(self):
        if Path(str(self.file_path)).suffix.lower() == '.pdf':
            print("in if")
            return PyMuPDFLoader(str(self.file_path))
        else:

            print("in else")
            return UnstructuredFileLoader(str(self.file_path), mode="elements",autodetect_encoding=True)

    def get_documents_from_file_by_path(self) -> Data:
        file_path = Path(str(self.file_path))
        file_name = file_path.name
        if file_path.exists():


            logging.info(f'file {file_name} processing')
            # loader = PyPDFLoader(str(file_path))
            file_extension = file_path.suffix.lower()
            try:
                loader = self.load_document_content()
                
                if file_extension == ".pdf":
                    pages = loader.load()
                    print("after load_document_content")
                else:
                    unstructured_pages = loader.load()   
                    pages = self.get_pages_with_page_numbers(unstructured_pages)      

            except Exception as e:
                raise Exception('Error while reading the file content or metadata')
        else:
            logging.info(f'File {file_name} does not exist')
            raise Exception(f'File {file_name} does not exist')
        print("before data")
        data = Data(data={"file_name": file_name, "pages": pages, "file_extension": file_extension})
        print("after data")
        # self.status = data if data else "No data"
        print("after status")
        print(data)
        return data or Data()

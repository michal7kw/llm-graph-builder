# copy of the PDFComponent but with the get_documents_from_file_by_path method (fille: local_file.py)
# extract_graph_from_file_local_file() --> get_documents_from_file_by_path()

# this component is to load pdf files and return the pages as a list of documents
# Status: working

from pathlib import Path
from langflow.custom import Component
from langflow.io import FileInput, BoolInput, Output
from langflow.schema import Data
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import logging

class PDFComponent(Component):
    display_name = "PDF"
    description = "PDFComponent_reimplementation_simplified"
    icon = "file-text"

    inputs = [
        FileInput(
            name="path",
            display_name="Path",
            file_types=["pdf"],
            info="Supported file types: pdf",
        ),
        BoolInput(
            name="silent_errors",
            display_name="Silent Errors",
            advanced=True,
            info="If true, errors will not raise an exception.",
        ),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="load_file"),
    ]

    def load_file(self) -> Data:
        logging.info("Starting load_file method")
        try:
            if not self.path:
                raise ValueError("Please upload a PDF to use this component.")
            
            resolved_path = Path(self.resolve_path(self.path))
            silent_errors = self.silent_errors

            file_extension = resolved_path.suffix.lower()
            if file_extension != ".pdf":
                raise ValueError(f"Unsupported file type: {file_extension}")

            file_name = resolved_path.name

            logging.info(f"Loading PDF file: {resolved_path}")
            loader = PyMuPDFLoader(str(resolved_path))
            pages = loader.load()
            
            # Convert pages to the format you want
            formatted_pages = [
                Document(page_content=page.page_content, metadata={
                    'source': page.metadata.get('source'),
                    'page_number': page.metadata.get('page_number'),
                    'filename': file_name,
                    'filetype': 'pdf'
                })
                for page in pages
            ]

            data = Data(data={
                "file_name": file_name,
                "pages": formatted_pages,
                "file_extension": file_extension
            })
            
            # self.status = "Data loaded successfully" if data else "No data"
            
            logging.info(f"Loaded PDF '{file_name}' with {len(formatted_pages)} pages")
            print(data)
            return data

        except Exception as e:
            error_msg = f"Error in load_file: {str(e)}"
            logging.error(error_msg)
            if not silent_errors:
                raise
            return Data()

        finally:
            logging.info("Exiting load_file method")
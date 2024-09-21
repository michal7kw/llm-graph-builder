from pathlib import Path

from langflow.base.data.utils import parse_text_file_to_data
from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from langflow.custom import Component
from langflow.io import BoolInput, FileInput, Output
from langflow.schema import Data


class PDFComponent(Component):
    display_name = "PDF"
    description = "Default A PDF file loader."
    icon = "file-text"

    inputs = [
        FileInput(
            name="path",
            display_name="Path Loader Default",
            file_types=["pdf"],
            info=f"Supported file types: pdf",
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
        if not self.path:
            raise ValueError("Please, upload a PDF to use this component.")
        resolved_path = self.resolve_path(self.path)
        silent_errors = self.silent_errors

        extension = Path(resolved_path).suffix[1:].lower()

        if extension != "pdf":
            raise ValueError(f"Unsupported file type: {extension}")

        loader = PyMuPDFLoader(str(self.path))
        text = " ".join(document.page_content for document in loader.load())
        data = Data(data={"file_path": str(self.path), "text": text})
        self.status = data if data else "No data"
        return data or Data()

from pathlib import Path
from typing import List, Union
from tqdm.auto import tqdm
import fitz
from langchain.text_splitter import CharacterTextSplitter
from loguru import logger

class DocumentChunker:
    def __init__(self, overlap_size: int = 200) -> None:
        self.overlap_size = overlap_size

    def chunk_document(
        self, file_path: Union[str, Path], max_chunk_size: int, **kwargs
    ) -> List[dict]:

        logger.info(f"Processing document: {file_path}")

        chunks = []
        text_splitter = CharacterTextSplitter(
            separator="\n",
            keep_separator=True,
            chunk_size=max_chunk_size,
            chunk_overlap=self.overlap_size,
        )

        document = fitz.open(file_path)
        accumulated_text = ""
        for page in document:
            page_text = page.get_text("block")

            if len(page_text) > max_chunk_size:
                chunks.append(
                    {"text": accumulated_text, "metadata": {"page": page.number}}
                )
                split_chunks = text_splitter.split_text(page_text)
                for chunk in split_chunks:
                    logger.info(
                        f"Saving chunk. Length: {len(chunk)}, page: {page.number}"
                    )
                    chunks.append(
                        {"text": chunk, "metadata": {"page": page.number}}
                    )
                accumulated_text = ""

            elif len(accumulated_text + page_text) >= max_chunk_size:
                if accumulated_text != "":
                    chunks.append(
                        {"text": accumulated_text, "metadata": {"page": page.number}}
                    )
                logger.info(
                    f"Saving chunk. Length: {len(accumulated_text)}, page: {page.number}"
                )
                accumulated_text = page_text

            else:
                accumulated_text += page_text

        chunks = [
            chunk for chunk in chunks if chunk["text"].strip().replace(" ", "")
        ]
        return chunks

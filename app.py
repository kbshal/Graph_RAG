from modules.data_embedding import TextEmbedder
from modules.local_llm import KnowledgeGraphLLM
from vectore_storage.weaviate_handler import VectorDBManager

from helpers.pdf_processor import pdf_processor
from helpers.graph_builder import ConceptGraph

import torch

if __name__ == '__main__':
    chunker = DocumentChunker()
    doc_chunks = chunker.chunk_document(r"/Users/bishalkharal/mero_space/Documents/my_resume.pdf", max_chunk_size=500)

    rag_model = RAGModel(
        model_directory="/Users/bishalkharal/Downloads/mistral-7b-orca",
        temperature=1.0,
        top_k=5,
        top_p=0.8,
        top_a=0.9,
        token_repetition_penalty=1.2
    )
    rag_model.setup_model()
    extracted_entities_df = rag_model.extract_entities(doc_chunks, max_new_tokens=1000)

    concept_graph = ConceptGraph(extracted_entities_df)
    concept_graph.build_graph()

    embedder_model = TextEmbedder()
    entity_embeddings = embedder_model.embed_texts(list(extracted_entities_df['node_1'] + ' ' + extracted_entities_df['node_2'] + ' ' + extracted_entities_df['edge']))

    extracted_entities_df['vectors'] = entity_embeddings.tolist()

    vector_db = VectorDBManager(db_name='Demo')
    vector_db.upload_vectors(extracted_entities_df)

    search_query = "what projects he did in AI/ML"
    search_results = vector_db.search_by_keyword(query=search_query, top_k=5)

    for result in search_results['data']['Get']['Demo']:
        concept_graph.extract_subgraph(result['source'])
    
    rag_model.generate_responses(doc_chunks, search_query, 500)

import sys
import os
from loguru import logger
from helpers.pdf_processor import PDFProcessor

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import re
import pandas as pd


def convert_keys_to_lowercase(dictionary):
    return {key: value.lower() for key, value in dictionary.items()}


def refine_results(data_list):
    pruned_list = [entry for entry in data_list if len(entry) > 1]

    exclude_elements = {
        'node_1': 'A concept from extracted ontology',
        'node_2': 'A related concept from extracted ontology',
        'edge': 'relationship between the two concepts, node_1 and node_2 in one or two sentences'
    }

    pruned_list = [convert_keys_to_lowercase(entry) for entry in pruned_list if entry != exclude_elements]
    return pruned_list


ontology_extraction_prompt = """
You are a network graph maker who extracts terms and their relations from a given context. 
You are provided with a context chunk (delimited by ```) Your task is to extract the ontology
of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n
Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n
\tTerms may include object, entity, location, organization, person, \n
\tcondition, acronym, documents, service, concept, etc.\n
\tTerms should be as atomistic as possible\n\n
Thought 2: Think about how these terms can have one on one relation with other terms.\n
\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n
\tTerms can be related to many other terms\n\n
Thought 3: Find out the relation between each such related pair of terms. \n\n
Format your output as a list of json. Each element of the list contains a pair of terms
and the relation between them, like the following: \n
[\n
    {\n
        "node_1": "A concept from extracted ontology",\n
        "node_2": "A related concept from extracted ontology",\n
        "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n
    }, {...}\n"
]
DO NOT RETURN ANY EXPLANATION, ONLY RETURN THE LIST OF JSON.
"""

qa_assistant_prompt = """You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'.
You only respond once as Assistant. You are allowed to use only the given context below to answer the user's queries, 
and if the answer is not present in the context, say you don't know the answer.
CONTEXT: {context}
"""

class KnowledgeGraphLLM:
    def __init__(self, model_path: str, temp: float, top_k_val: float, top_p_val: float, top_a_val: float, repetition_penalty: float):
        self.model_path = model_path
        self.temp = temp
        self.top_k_val = top_k_val
        self.top_p_val = top_p_val
        self.top_a_val = top_a_val
        self.repetition_penalty = repetition_penalty

    def initialize_model(self) -> None:
        self.config = ExLlamaV2Config()
        self.config.model_dir = self.model_path
        self.config.prepare()

        self.model = ExLlamaV2(self.config)
        logger.info("Loading model...")

        self.cache = ExLlamaV2Cache(self.model, lazy=True)
        self.model.load_autosplit(self.cache)

        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)

        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.temperature = self.temp
        self.settings.top_k = self.top_k_val
        self.settings.top_p = self.top_p_val
        self.settings.top_a = self.top_a_val
        self.settings.token_repetition_penalty = self.repetition_penalty
        self.settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

    def extract_nodes(self, text_chunks, max_tokens) -> str:
        if not self.generator or not self.settings:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        results = []

        self.generator.warmup()

        prompt_template = """system
        {system_prompt}
        
        user
        {text_chunk}
        
        assistant
        """

        for idx, chunk in enumerate(text_chunks):
            logger.info(f"Extracting tuples from chunk: {idx}")
            output = self.generator.generate_simple(prompt_template.format(system_prompt=ontology_extraction_prompt, text_chunk=chunk['text']), self.settings, max_tokens, seed=1234)
            
            # Extracting dict types
            pattern = r'\{[^}]+\}'
            matches = re.findall(pattern, output)
            try:
                dicts = [eval(match) for match in matches]
                dicts = refine_results(dicts)

                for entry in dicts:
                    entry['chunk'] = chunk['text']
                    
                results.extend(dicts)
            except:
                pass
        
        df = pd.DataFrame(results)
        df = df.drop_duplicates(subset=['node_1', 'node_2', 'edge'], keep=False)
        return df
    
    def generate_responses(self, text_chunks, query, max_tokens) -> str:
        if not self.generator or not self.settings:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        self.generator.warmup()

        prompt_template = """system
        {qa_prompt}
        
        user
        {query}
        
        assistant
        """
        logger.info(f"Asking the assistant: {query}")
        output = self.generator.generate_simple(prompt_template.format(qa_prompt=qa_assistant_prompt.format(context=text_chunks), query=query), self.settings, max_tokens, seed=1234)

        start_tag = ""
        end_tag = ""
        start_index = output.rfind(start_tag)
        end_index = output.rfind(end_tag)
        logger.info(f"Answer: {output[start_index + len(start_tag): end_index]}")

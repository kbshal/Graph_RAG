import weaviate
import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class VectorDBManager:
    def __init__(self, db_name: str) -> None:
        self.db_name = db_name
        auth_config = weaviate.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])

        self.client = weaviate.Client(
            url=os.environ['WEAVIATE_CLIENT_URL'],
            auth_client_secret=auth_config
        )

        schema_config = {
            "class": self.db_name,
            "vectorizer": "none",
        }
        self.client.schema.create_class(schema_config)

        logger.info('Database initialized')

    def upload_vectors(self, data):
        self.client.batch.configure(batch_size=100)
        with self.client.batch as batch:
            for i, row in data.iterrows():
                print(f"Uploading entry: {i + 1}")

                properties = {
                    "source": row["node_1"],
                    "relation": row["edge"],
                    "target": row["node_2"],
                    "chunk": row["chunk"]
                }

                batch.add_data_object(properties, self.db_name, vector=row["vectors"])

        total_items = self.client.query.aggregate(self.db_name).with_meta_count().do()
        logger.info(f'Total items in database: {total_items}')

    def search_by_keyword(self, query: str, top_k: int = 5):
        search_results = (
            self.client.query.get(self.db_name, ["source", "target", "relation", "chunk"])
            .with_bm25(query=query)
            .with_limit(top_k)
            .do()
        )
        return search_results

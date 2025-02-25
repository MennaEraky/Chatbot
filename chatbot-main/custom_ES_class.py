from langchain_elasticsearch import ElasticsearchChatMessageHistory

class CustomElasticsearchChatMessageHistory(ElasticsearchChatMessageHistory):
    def save_order_id(self, session_id: str, order_id: str):
        self.client.update_by_query(
            index=self.index,
            body={
                "script": {
                    "source": """
                    if (!ctx._source.containsKey('order_id')) {
                        ctx._source.order_id = params.order_id;
                    }
                    """,
                    "params": {"order_id": order_id}
                },
                "query": {
                    "term": {"session_id": session_id}
                }
            }
        )

    def get_order_id(self, session_id: str) -> str:
        response = self.client.search(
            index=self.index,
            body={
                "query": {
                    "term": {"session_id": session_id}
                },
                "size": 1 
            }
        )
        if response["hits"]["hits"]:
            return response["hits"]["hits"][0]["_source"].get("order_id", None)
        return None

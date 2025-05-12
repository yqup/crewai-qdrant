from crewai.tools import BaseTool
from pydantic import BaseModel, Field, root_validator, PrivateAttr
import os
from typing import Optional, List, Dict, Any, Callable
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid
import openai

class QdrantInput(BaseModel):
    action: str = Field(
        ...,
        description="Action to perform: 'add', 'update', 'list', 'delete', 'search'"
    )
    collection_name: str = Field(
        ...,
        description="Name of the Qdrant collection to operate on"
    )
    content: Optional[str] = Field(
        None,
        description="Content to add or update (required for add/update actions)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata associated with the content (optional for add/update actions)"
    )
    point_id: Optional[str] = Field(
        None,
        description="ID of the point to update or delete (required for update/delete actions)"
    )
    query: Optional[str] = Field(
        None,
        description="Search query (required for search action)"
    )
    filter_by: Optional[str] = Field(
        None,
        description="Metadata field to filter on (for search action)"
    )
    filter_value: Optional[str] = Field(
        None,
        description="Value to filter by for the specified field (for search action)"
    )
    limit: Optional[int] = Field(
        10,
        description="Maximum number of results to return (for list/search actions)"
    )

    @root_validator(pre=True)
    def validate_action_requirements(cls, values):
        action = values.get("action")
        
        if action in ["add", "update"] and not values.get("content"):
            raise ValueError(f"Content is required for {action} action")
            
        if action in ["update", "delete"] and not values.get("point_id"):
            raise ValueError(f"Point ID is required for {action} action")
            
        if action == "search" and not values.get("query"):
            raise ValueError("Query is required for search action")
            
        return values

class QdrantToolInputWrapper(BaseModel):
    input: QdrantInput

class QdrantTool(BaseTool):
    name: str = "Qdrant Content Management Tool"
    description: str = (
        "A tool for managing content in Qdrant vector database. "
        "Supports adding, updating, listing, and deleting content with metadata. "
        "Uses OpenAI's text-embedding-3-small model for vectorization by default."
    )
    args_schema: type[BaseModel] = QdrantInput
    _client: QdrantClient = PrivateAttr()
    __custom_embedding_fn: Optional[Callable[[str], List[float]]] = PrivateAttr(default=None)

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        cluster_id: Optional[str] = None,
        custom_embedding_fn: Optional[Callable[[str], List[float]]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._client = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL", "https://e23f69dd-4ac1-4c56-afd0-f9c98a474544.eu-west-2-0.aws.cloud.qdrant.io"),
            api_key=qdrant_api_key or os.getenv("QDRANT_API_KEY")
        )
        self.__custom_embedding_fn = custom_embedding_fn

    def _run(
        self,
        action: str,
        collection_name: str,
        content: str = None,
        metadata: dict = None,
        point_id: str = None,
        query: str = None,
        filter_by: str = None,
        filter_value: str = None,
        limit: int = 10
    ) -> str:
        try:
            # Ensure collection exists
            if not self._client.collection_exists(collection_name):
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )

            if action == "add":
                return self._add_content(collection_name, content, metadata)
            elif action == "update":
                return self._update_content(collection_name, point_id, content, metadata)
            elif action == "list":
                return self._list_content(collection_name, limit)
            elif action == "delete":
                return self._delete_content(collection_name, point_id)
            elif action == "search":
                return self._search_content(
                    collection_name,
                    query,
                    limit,
                    filter_by,
                    filter_value
                )
            else:
                return f"Unknown action '{action}'. Valid actions are: add, update, list, delete, search."

        except Exception as e:
            return f"Error: {str(e)}"

    def _add_content(self, collection_name: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        point_id = str(uuid.uuid4())
        point = PointStruct(
            id=point_id,
            vector=self._get_embedding(content),
            payload={"content": content, "metadata": metadata or {}}
        )
        self._client.upsert(collection_name=collection_name, points=[point])
        return f"Content added successfully with ID: {point_id}"

    def _update_content(self, collection_name: str, point_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        point = PointStruct(
            id=point_id,
            vector=self._get_embedding(content),
            payload={"content": content, "metadata": metadata or {}}
        )
        self._client.upsert(collection_name=collection_name, points=[point])
        return f"Content updated successfully for ID: {point_id}"

    def _list_content(self, collection_name: str, limit: int = 10) -> str:
        points = self._client.scroll(
            collection_name=collection_name,
            limit=limit
        )[0]
        
        if not points:
            return "No content found in collection."
            
        result = []
        for point in points:
            result.append(f"ID: {point.id}")
            result.append(f"Content: {point.payload.get('content', 'N/A')}")
            result.append(f"Metadata: {point.payload.get('metadata', {})}")
            result.append("---")
            
        return "\n".join(result)

    def _delete_content(self, collection_name: str, point_id: str) -> str:
        self._client.delete(
            collection_name=collection_name,
            points_selector=[point_id]
        )
        return f"Content deleted successfully for ID: {point_id}"

    def _search_content(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        filter_by: Optional[str] = None,
        filter_value: Optional[str] = None
    ) -> str:
        # Build search filter if metadata filtering is requested
        search_filter = None
        if filter_by and filter_value:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key=f"metadata.{filter_by}",
                        match=MatchValue(value=filter_value)
                    )
                ]
            )

        search_result = self._client.search(
            collection_name=collection_name,
            query_vector=self._get_embedding(query),
            limit=limit,
            query_filter=search_filter
        )
        
        if not search_result:
            return "No matching content found."
            
        result = []
        for point in search_result:
            result.append(f"ID: {point.id}")
            result.append(f"Content: {point.payload.get('content', 'N/A')}")
            result.append(f"Metadata: {point.payload.get('metadata', {})}")
            result.append(f"Score: {point.score}")
            result.append("---")
            
        return "\n".join(result)

    def _get_embedding(self, text: str) -> List[float]:
        if self.__custom_embedding_fn:
            return self.__custom_embedding_fn(text)
            
        # Default to OpenAI's text-embedding-3-small model
        client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding 
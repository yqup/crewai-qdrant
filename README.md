# CrewAI Qdrant Tool

A CrewAI tool for seamlessly managing and searching vector embeddings in Qdrant. This tool integrates with OpenAI's text-embedding-3-small model by default for vector embeddings and is designed to work directly as a tool in CrewAI tasks and agents.

**Author**: Tony Wood (tony@tonywood.org)

> For official documentation, visit [CrewAI Qdrant Vector Search Tool Documentation](https://docs.crewai.com/tools/qdrantvectorsearchtool)
> 
> For understanding CrewAI tools in general, see [CrewAI Tools Core Concepts](https://docs.crewai.com/concepts/tools)

## What is a CrewAI Tool?

A tool in CrewAI is a skill or function that agents can utilize to perform various actions. This Qdrant tool is part of the CrewAI toolkit, enabling agents to work with vector embeddings and semantic search capabilities. It follows CrewAI's tool design principles:

- **Utility**: Crafted for vector similarity search and content management
- **Integration**: Seamlessly integrates into agent workflows
- **Customizability**: Supports custom embedding functions
- **Error Handling**: Includes robust error handling mechanisms
- **Caching Mechanism**: Features intelligent caching for optimized performance

## Features

- **Direct CrewAI Integration**: Plug-and-play as a tool in CrewAI tasks and agents
- **Content Management**: Add, update, list, and delete content in Qdrant collections
- **Vector Search**: Perform semantic searches using vector embeddings
- **Metadata Support**: Associate and filter content using custom metadata
- **Flexible Configuration**: Use custom embedding functions or default to OpenAI's embeddings
- **Easy Integration**: Seamlessly works with CrewAI's tool ecosystem

## Installation

```bash
pip install crewai
pip install 'crewai[tools]'  # For additional CrewAI tools support
```

## Environment Variables

The tool requires the following environment variables:

- `QDRANT_URL` - Your Qdrant instance URL
- `QDRANT_API_KEY` - Your Qdrant API key
- `OPENAI_API_KEY` - Your OpenAI API key (if using default embeddings)

## Usage

### Basic Setup with CrewAI

```python
from crewai import Agent, Task, Crew
from crewai_qdrant import QdrantTool

# Initialize the tool
qdrant_tool = QdrantTool(
    qdrant_url="your-qdrant-url",  # Optional if set in env
    qdrant_api_key="your-api-key"  # Optional if set in env
)

# Create an agent with the tool
agent = Agent(
    role="Research Assistant",
    goal="Search and analyze vector embeddings",
    tools=[qdrant_tool],
    verbose=True
)

# Use in a task
task = Task(
    description="Search for relevant information about X",
    agent=agent
)

# Create a crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

# Execute the task
result = crew.kickoff()
```

### Supported Operations that the tool will run

1. **Add Content**
```python
result = qdrant_tool.run(
    action="add",
    collection_name="my_collection",
    content="Your content here",
    metadata={"category": "example", "tags": ["demo"]}
)
```

2. **Update Content**
```python
result = qdrant_tool.run(
    action="update",
    collection_name="my_collection",
    point_id="existing-id",
    content="Updated content",
    metadata={"category": "updated"}
)
```

3. **List Content**
```python
result = qdrant_tool.run(
    action="list",
    collection_name="my_collection",
    limit=10
)
```

4. **Delete Content**
```python
result = qdrant_tool.run(
    action="delete",
    collection_name="my_collection",
    point_id="content-id"
)
```

5. **Search Content**
```python
result = qdrant_tool.run(
    action="search",
    collection_name="my_collection",
    query="search query",
    filter_by="category",  # Optional
    filter_value="example",  # Optional
    limit=5
)
```

### Custom Embedding Function

You can provide your own embedding function:

```python
def custom_embeddings(text: str) -> List[float]:
    # Your custom embedding logic here
    return vector_embedding

qdrant_tool = QdrantTool(
    custom_embedding_fn=custom_embeddings
)
```

## Error Handling

The tool includes built-in error handling and validation:
- Validates required parameters for each action
- Ensures collections exist before operations
- Returns descriptive error messages
- Gracefully handles API and connection errors

## Caching

The tool implements CrewAI's caching mechanism to optimize performance:
- Caches search results to reduce redundant API calls
- Maintains vector embedding cache for frequently used content
- Optimizes performance for repeated operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[License information here]

## Documentation

For more detailed information and advanced usage examples, please refer to:
- [Official CrewAI Qdrant Vector Search Tool documentation](https://docs.crewai.com/tools/qdrantvectorsearchtool)
- [CrewAI Tools Core Concepts](https://docs.crewai.com/concepts/tools)


from langgraph.store.memory import InMemoryStore


# 1. Create long-term memory store
store = InMemoryStore()


# 2. Define user namespace
# In a real application, use a stable authenticated user ID.
namespace = ("user_001", "pipeline_memories")


# 3. Save memory
store.put(
    namespace,
    "customer_etl_error",
    {
        "pipeline": "customer_etl",
        "last_error": "Database connection timeout",
        "recommendation": "Check database connectivity"
    }
)

print("Memory saved successfully!")


# 4. Retrieve memory
memory = store.get(
    namespace,
    "customer_etl_error"
)

if memory:
    print("\nRetrieved Long-term Memory:")
    print(memory.value)


# 5. Save another memory
store.put(
    namespace,
    "sales_etl_error",
    {
        "pipeline": "sales_etl",
        "last_error": "Out of memory",
        "recommendation": "Review Spark executor memory"
    }
)


# 6. Search memories in the same namespace
all_memories = store.search(namespace)

print("\nAll Stored Memories:")

for item in all_memories:
    print(item.key, ":", item.value)

"""Kafka topic and index names from the case study."""

QUERY_LOGS = "query_logs"
LOG_PROCESSING_QUEUE = "log_processing_queue"
QUERY_PROCESS_QUEUE = "query_process_queue"
SEGMENTED_QUERIES_QUEUE = "segmented_queries_queue"
LIST_UPDATE = "list_update"
CATALOG_UPDATES = "catalog_updates"

ALL_TOPICS = [
    QUERY_LOGS,
    LOG_PROCESSING_QUEUE,
    QUERY_PROCESS_QUEUE,
    SEGMENTED_QUERIES_QUEUE,
    LIST_UPDATE,
    CATALOG_UPDATES,
]

PRODUCT_SEARCH_INDEX = "product_search_index"
SEMANTIC_SEARCH_INDEX = "semantic_search_index"
WARM_QUERY_CACHE_INDEX = "warm_query_cache"

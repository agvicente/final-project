"""
Kafka utility functions for topic management and isolation.

This module provides utilities for cleaning and managing Kafka topics
during experiments to ensure isolation between runs.
"""

import logging
import time
from typing import List

from kafka import KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import UnknownTopicOrPartitionError


# Configure logger
logger = logging.getLogger(__name__)


def purge_kafka_topics(
    bootstrap_servers: str = "localhost:9092",
    topics: List[str] = None
) -> bool:
    """
    Purge Kafka topics by deleting and recreating them.

    This function ensures complete isolation between experiment runs by
    removing all existing messages from the specified topics.

    Args:
        bootstrap_servers: Kafka broker address (default: "localhost:9092")
        topics: List of topic names to purge (default: ["packets", "flows", "alerts"])

    Returns:
        True if purge succeeded, False if non-critical error occurred but safe to continue

    Raises:
        SystemExit: If Kafka broker is unreachable (critical error)
    """
    # Use default topics if None provided
    if topics is None:
        topics = ["packets", "flows", "alerts"]

    client = None

    try:
        # Create admin client
        client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id="purge_kafka_topics"
        )

        # Delete topics
        try:
            client.delete_topics(topics)
            logger.info(f"Deleted topics: {topics}")
            # Wait for deletion to complete
            time.sleep(2)
        except UnknownTopicOrPartitionError:
            # Topic doesn't exist - that's fine, we'll create it
            logger.warning(f"Some topics don't exist yet: {topics}")
        except Exception as e:
            # Non-critical error during deletion
            logger.warning(f"Error during topic deletion: {e}")
            return False

        # Create topics
        try:
            new_topics = [
                NewTopic(name=topic, num_partitions=1, replication_factor=1)
                for topic in topics
            ]
            client.create_topics(new_topics)
            logger.info(f"Created topics: {topics}")
        except Exception as e:
            # Non-critical error during creation
            logger.warning(f"Error during topic creation: {e}")
            return False

        return True

    except Exception as e:
        # Critical error - Kafka unreachable
        logger.error(f"Critical error: Cannot connect to Kafka at {bootstrap_servers}: {e}")
        raise SystemExit(f"Kafka broker unreachable: {e}")

    finally:
        # Always close client
        if client is not None:
            client.close()

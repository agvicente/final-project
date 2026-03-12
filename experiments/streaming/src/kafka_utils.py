"""
Kafka utility functions for topic management and isolation.

This module provides utilities for cleaning and managing Kafka topics
during experiments to ensure isolation between runs.

Includes synchronization utilities to coordinate multi-stage pipelines
(producer → flow_consumer → detector).
"""

import logging
import time
from typing import List, Optional

from kafka import KafkaAdminClient, KafkaConsumer, TopicPartition
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


def get_topic_end_offset(
    topic: str,
    bootstrap_servers: str = "localhost:9092",
) -> int:
    """
    Get the total end offset (high watermark) of a topic across all partitions.

    Args:
        topic: Topic name
        bootstrap_servers: Kafka broker address

    Returns:
        Sum of end offsets across all partitions
    """
    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        consumer_timeout_ms=5000,
    )
    try:
        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            return 0

        total = 0
        for p in partitions:
            tp = TopicPartition(topic, p)
            consumer.assign([tp])
            consumer.seek_to_end(tp)
            total += consumer.position(tp)
        return total
    finally:
        consumer.close()


def wait_for_flow_consumer(
    bootstrap_servers: str = "localhost:9092",
    flows_topic: str = "flows",
    stable_seconds: float = 5.0,
    poll_interval: float = 1.0,
    timeout_seconds: float = 300.0,
) -> int:
    """
    Wait until the FlowConsumer finishes producing flows.

    Monitors the 'flows' topic end offset. When the offset stops growing
    for `stable_seconds`, we consider the FlowConsumer done.

    This avoids the race condition where the detector starts before all
    packets have been aggregated into flows.

    Args:
        bootstrap_servers: Kafka broker address
        flows_topic: Topic where FlowConsumer publishes flows
        stable_seconds: Seconds of no growth before considering done
        poll_interval: Seconds between offset checks
        timeout_seconds: Maximum wait time

    Returns:
        Final number of flows in topic

    Raises:
        TimeoutError: If timeout_seconds exceeded without stabilization
    """
    start = time.time()
    last_offset = -1
    stable_since: Optional[float] = None

    logger.info(f"Aguardando FlowConsumer estabilizar (topico '{flows_topic}')...")

    while True:
        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"FlowConsumer nao estabilizou em {timeout_seconds}s. "
                f"Ultimo offset: {last_offset}"
            )

        current_offset = get_topic_end_offset(flows_topic, bootstrap_servers)

        if current_offset != last_offset:
            # Still growing
            last_offset = current_offset
            stable_since = time.time()
            logger.info(
                f"  flows: {current_offset} (crescendo, {elapsed:.0f}s)"
            )
        elif stable_since is not None:
            stable_elapsed = time.time() - stable_since
            if stable_elapsed >= stable_seconds:
                logger.info(
                    f"  flows: {current_offset} (estavel por {stable_elapsed:.0f}s) — pronto"
                )
                return current_offset
        else:
            # First poll, no flows yet
            stable_since = time.time()

        time.sleep(poll_interval)

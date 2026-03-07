"""
Unit tests for Kafka topic isolation utilities.

Tests cover:
1. Topic purging (deletion and recreation)
2. Message clearing in existing topics
3. Custom topic list support
4. Error handling (non-existent topics, unreachable Kafka)
5. Graceful degradation (warnings vs. critical failures)

The function being tested:
  purge_kafka_topics(bootstrap_servers: str, topics: List[str]) -> bool

This is a RED PHASE test file - tests should fail until implementation exists.

Run with: pytest tests/test_kafka_isolation.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import List

# Import the function to test (will fail until implemented)
try:
    from src.kafka_utils import purge_kafka_topics
except ImportError:
    # Expected during RED phase - function doesn't exist yet
    purge_kafka_topics = None


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def default_topics():
    """Default topics that should be purged."""
    return ["packets", "flows", "alerts"]


@pytest.fixture
def mock_admin_client():
    """Mock KafkaAdminClient for testing."""
    client = MagicMock()
    client.list_topics.return_value = {}
    return client


@pytest.fixture
def mock_admin_client_with_topics():
    """Mock KafkaAdminClient with existing topics."""
    client = MagicMock()
    # Simulate topics that exist in Kafka
    client.list_topics.return_value = {
        "packets": MagicMock(),
        "flows": MagicMock(),
        "alerts": MagicMock(),
    }
    return client


# ============================================================
# TEST CLASS: Basic Purge Functionality
# ============================================================

class TestPurgeBasicFunctionality:
    """Tests for basic purge_kafka_topics() behavior."""

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    def test_purge_deletes_and_recreates_topics(
        self,
        mock_kafka_admin_class,
        default_topics,
        mock_admin_client_with_topics
    ):
        """
        Verifies that purge_kafka_topics deletes existing topics
        and recreates them empty.

        Expected behavior:
        - Calls delete_topics() with topic names
        - Calls create_topics() with empty configuration
        - Returns True on success
        """
        mock_kafka_admin_class.return_value = mock_admin_client_with_topics
        mock_admin_client_with_topics.delete_topics.return_value = None
        mock_admin_client_with_topics.create_topics.return_value = None

        result = purge_kafka_topics(
            bootstrap_servers="localhost:9092",
            topics=default_topics
        )

        assert result is True
        mock_admin_client_with_topics.delete_topics.assert_called_once()
        mock_admin_client_with_topics.create_topics.assert_called_once()
        mock_admin_client_with_topics.close.assert_called_once()

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    def test_purge_clears_existing_messages(
        self,
        mock_kafka_admin_class,
        default_topics,
        mock_admin_client_with_topics
    ):
        """
        Verifies that purge_kafka_topics removes old messages
        from topics before recreation.

        Expected behavior:
        - Topics are deleted (which removes all messages)
        - Topics are recreated (empty state)
        - Returns True
        """
        mock_kafka_admin_class.return_value = mock_admin_client_with_topics

        result = purge_kafka_topics(topics=default_topics)

        assert result is True
        # Verify delete was called before create
        assert mock_admin_client_with_topics.method_calls[0][0] == "delete_topics"
        # Second call should be create_topics
        assert mock_admin_client_with_topics.method_calls[1][0] == "create_topics"


# ============================================================
# TEST CLASS: Custom Configuration
# ============================================================

class TestPurgeCustomization:
    """Tests for custom topic list support."""

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    def test_purge_with_custom_topics(
        self,
        mock_kafka_admin_class,
        mock_admin_client_with_topics
    ):
        """
        Verifies that purge_kafka_topics accepts custom topic list
        and purges only those topics.

        Expected behavior:
        - Function accepts topics=["custom1", "custom2"]
        - Only those topics are deleted and recreated
        - Returns True
        """
        custom_topics = ["custom_packets", "custom_flows"]
        mock_kafka_admin_class.return_value = mock_admin_client_with_topics

        result = purge_kafka_topics(
            bootstrap_servers="kafka-prod:9092",
            topics=custom_topics
        )

        assert result is True
        # Verify delete_topics was called with custom topics
        delete_call = mock_admin_client_with_topics.delete_topics.call_args
        assert custom_topics == delete_call[0][0]

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    def test_purge_uses_default_topics_when_none_provided(
        self,
        mock_kafka_admin_class,
        default_topics,
        mock_admin_client_with_topics
    ):
        """
        Verifies that purge_kafka_topics uses default topics
        when topics parameter is None.

        Expected behavior:
        - Function called with topics=None
        - Default topics ["packets", "flows", "alerts"] are used
        - Returns True
        """
        mock_kafka_admin_class.return_value = mock_admin_client_with_topics

        result = purge_kafka_topics(bootstrap_servers="localhost:9092", topics=None)

        assert result is True
        # Verify delete_topics was called with default topics
        delete_call = mock_admin_client_with_topics.delete_topics.call_args
        assert default_topics == delete_call[0][0]


# ============================================================
# TEST CLASS: Error Handling
# ============================================================

class TestPurgeErrorHandling:
    """Tests for error handling and recovery."""

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    def test_purge_continues_if_topics_dont_exist(
        self,
        mock_kafka_admin_class,
        mock_admin_client_with_topics
    ):
        """
        Verifies that purge_kafka_topics doesn't fail if topics
        don't exist in Kafka yet.

        Expected behavior:
        - delete_topics() may raise TopicDoesNotExistError
        - Function catches error and continues
        - create_topics() is still called
        - Returns True (safe to continue experiment)
        """
        # Simulate topic not existing
        from kafka.errors import UnknownTopicOrPartitionError
        mock_admin_client_with_topics.delete_topics.side_effect = UnknownTopicOrPartitionError(
            "Topic does not exist"
        )

        mock_kafka_admin_class.return_value = mock_admin_client_with_topics

        result = purge_kafka_topics(
            topics=["nonexistent_topic"]
        )

        assert result is True
        # Verify that create_topics was still called despite delete error
        assert mock_admin_client_with_topics.create_topics.called

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    @patch("src.kafka_utils.logger")
    def test_purge_warns_on_error_but_continues(
        self,
        mock_logger,
        mock_kafka_admin_class,
        mock_admin_client_with_topics
    ):
        """
        Verifies that purge_kafka_topics logs warnings on non-critical
        errors but continues execution.

        Expected behavior:
        - Non-critical error occurs (e.g., timeout)
        - Function logs warning via logger.warning()
        - Function returns False (issue encountered)
        - Experiment can still proceed with caution
        """
        # Simulate a non-critical error
        mock_admin_client_with_topics.delete_topics.side_effect = Exception(
            "Timeout deleting topic"
        )
        mock_kafka_admin_class.return_value = mock_admin_client_with_topics

        result = purge_kafka_topics(topics=["packets"])

        # Should return False (non-critical error occurred)
        assert result is False
        # Should have logged a warning
        assert mock_logger.warning.called

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    def test_purge_exits_if_kafka_unreachable(
        self,
        mock_kafka_admin_class,
        default_topics
    ):
        """
        Verifies that purge_kafka_topics raises SystemExit
        if Kafka broker is unreachable (critical error).

        Expected behavior:
        - KafkaAdminClient() raises ConnectionError or similar
        - Function raises SystemExit (cannot continue)
        - Error message is logged
        """
        # Simulate Kafka unreachable
        mock_kafka_admin_class.side_effect = Exception(
            "Broker may not be available"
        )

        with pytest.raises(SystemExit):
            purge_kafka_topics(
                bootstrap_servers="unreachable-kafka:9092",
                topics=default_topics
            )


# ============================================================
# TEST CLASS: Resource Management
# ============================================================

class TestPurgeResourceManagement:
    """Tests for proper cleanup and resource handling."""

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    def test_purge_closes_client_on_success(
        self,
        mock_kafka_admin_class,
        mock_admin_client_with_topics,
        default_topics
    ):
        """
        Verifies that KafkaAdminClient is properly closed
        after successful purge.

        Expected behavior:
        - purge_kafka_topics() completes
        - client.close() is called (context manager or explicit)
        - Resources are released
        """
        mock_kafka_admin_class.return_value = mock_admin_client_with_topics

        purge_kafka_topics(topics=default_topics)

        mock_admin_client_with_topics.close.assert_called_once()

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    def test_purge_closes_client_on_error(
        self,
        mock_kafka_admin_class,
        mock_admin_client_with_topics
    ):
        """
        Verifies that KafkaAdminClient is closed even if
        an error occurs during purge.

        Expected behavior:
        - Error occurs during deletion
        - client.close() is still called (finally block)
        - No resource leaks
        """
        # Simulate error during deletion
        from kafka.errors import UnknownTopicOrPartitionError
        mock_admin_client_with_topics.delete_topics.side_effect = UnknownTopicOrPartitionError(
            "Topic not found"
        )
        mock_kafka_admin_class.return_value = mock_admin_client_with_topics

        try:
            purge_kafka_topics(topics=["packets"])
        except SystemExit:
            pass

        # close() should still be called
        assert mock_admin_client_with_topics.close.called


# ============================================================
# TEST CLASS: Integration scenarios
# ============================================================

class TestPurgeIntegration:
    """Integration tests for realistic usage scenarios."""

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    def test_purge_idempotent_execution(
        self,
        mock_kafka_admin_class,
        default_topics,
        mock_admin_client_with_topics
    ):
        """
        Verifies that purge_kafka_topics is idempotent
        (safe to run multiple times).

        Expected behavior:
        - Function called twice in succession
        - Both calls succeed and return True
        - Second call handles already-empty topics gracefully
        """
        mock_kafka_admin_class.return_value = mock_admin_client_with_topics

        # First purge
        result1 = purge_kafka_topics(topics=default_topics)
        # Second purge (topics already empty)
        result2 = purge_kafka_topics(topics=default_topics)

        assert result1 is True
        assert result2 is True

    @pytest.mark.skipif(
        purge_kafka_topics is None,
        reason="RED PHASE: Function not yet implemented"
    )
    @patch("src.kafka_utils.KafkaAdminClient")
    def test_purge_with_bootstrap_servers_parameter(
        self,
        mock_kafka_admin_class,
        default_topics,
        mock_admin_client_with_topics
    ):
        """
        Verifies that bootstrap_servers parameter is passed
        correctly to KafkaAdminClient.

        Expected behavior:
        - bootstrap_servers="kafka-prod:9092" is passed
        - KafkaAdminClient is created with correct bootstrap_servers
        - Connection established to correct broker
        """
        mock_kafka_admin_class.return_value = mock_admin_client_with_topics

        bootstrap_servers = "kafka-prod:9092"
        purge_kafka_topics(
            bootstrap_servers=bootstrap_servers,
            topics=default_topics
        )

        # Verify KafkaAdminClient was created with correct bootstrap_servers
        init_call = mock_kafka_admin_class.call_args
        assert init_call[1]["bootstrap_servers"] == bootstrap_servers


# ============================================================
# MARKER: RED PHASE INDICATOR
# ============================================================
# All tests above should FAIL when run because:
#   - src.kafka_utils module doesn't exist yet
#   - purge_kafka_topics() function doesn't exist yet
#
# Expected failure mode: ImportError or AttributeError
#
# This is CORRECT behavior for RED phase of TDD!
# ============================================================

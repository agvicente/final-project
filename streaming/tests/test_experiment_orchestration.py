"""
Unit tests for Experiment Orchestration with Unique Group IDs.

Tests follow TDD RED PHASE - all tests MUST FAIL because the feature
(unique group IDs per experiment execution) is not yet implemented.

Feature being tested:
- FlowConsumer: f"flow-consumer-{experiment_id}"
- StreamingDetector: f"detector-{experiment_id}"
- experiment_id format: datetime.now().strftime("%Y%m%d_%H%M%S_%f")

Run with: pytest tests/test_experiment_orchestration.py -v
"""

import pytest
import re
from datetime import datetime
from pathlib import Path


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary output directory for results."""
    return tmp_path / "results"


@pytest.fixture
def run_experiment_source():
    """Load the run_experiment.py source code."""
    script_path = Path(__file__).parent.parent / 'scripts' / 'run_experiment.py'
    with open(script_path, 'r') as f:
        return f.read()


# ============================================================
# HELPER: Extract group-id from code
# ============================================================

def extract_group_ids_from_source(source):
    """Extract all hardcoded group-id values from source code."""
    pattern = r'--group-id["\']?\s*,\s*["\']([^"\']+)["\']'
    return re.findall(pattern, source)


def extract_flow_consumer_group_id(source):
    """Extract the group-id used for FlowConsumer from source."""
    # Look for the pattern in start_flow_consumer function
    pattern = r'--group-id["\']?\s*,\s*["\']([^"\']+)["\']'
    # Focus on the start_flow_consumer function
    flow_consumer_func = source[source.find('def start_flow_consumer'):source.find('def start_flow_consumer') + 1000]
    matches = re.findall(pattern, flow_consumer_func)
    return matches[0] if matches else None


def extract_detector_group_id(source):
    """Extract the group_id used for StreamingDetector from source."""
    # Look for the pattern: group_id="..." or group_id='...'
    pattern = r'group_id\s*=\s*["\']([^"\']+)["\']'
    matches = re.findall(pattern, source)
    # Get the one from StreamingDetectorConfig context
    if matches:
        return matches[0]  # First occurrence should be in StreamingDetectorConfig
    return None


# ============================================================
# TEST 1: Experiment generates unique ID
# ============================================================

class TestExperimentGeneratesUniqueId:
    """
    TEST 1: test_experiment_generates_unique_id

    Verifies that FlowConsumer group-id is NOT the hardcoded value.
    This test FAILS when feature is not implemented (current state).

    EXPECTED FAILURE: Currently run_experiment.py uses fixed group ID
    "experiment-flow-processor" instead of unique ID.
    """

    def test_experiment_generates_unique_id(self, run_experiment_source):
        """Verify FlowConsumer does not use hardcoded group-id."""
        flow_consumer_id = extract_flow_consumer_group_id(run_experiment_source)

        # This assertion MUST FAIL - currently hardcoded
        assert flow_consumer_id is not None, "FlowConsumer must have group-id"
        assert flow_consumer_id != 'experiment-flow-processor', \
            f"FAIL: Group ID is hardcoded '{flow_consumer_id}'. Should be unique with timestamp format."


# ============================================================
# TEST 2: FlowConsumer uses unique group ID with correct pattern
# ============================================================

class TestFlowConsumerUsesUniqueGroupId:
    """
    TEST 2: test_flow_consumer_uses_unique_group_id

    Verifies that FlowConsumer command includes dynamic group-id
    matching pattern: flow-consumer-YYYYMMDD_HHMMSS_microseconds

    EXPECTED FAILURE: Currently uses hardcoded "experiment-flow-processor"
    """

    def test_flow_consumer_uses_unique_group_id(self, run_experiment_source):
        """FlowConsumer should use pattern: flow-consumer-<timestamp>"""
        flow_consumer_id = extract_flow_consumer_group_id(run_experiment_source)

        # This test MUST FAIL - pattern should be flow-consumer-YYYYMMDD_HHMMSS_microseconds
        pattern = r'^flow-consumer-\d{8}_\d{6}_\d{6}$'
        assert re.match(pattern, flow_consumer_id or ''), \
            f"FAIL: Group ID should match pattern '{pattern}', got: '{flow_consumer_id}'"


# ============================================================
# TEST 3: Detector uses unique group ID with correct pattern
# ============================================================

class TestDetectorUsesUniqueGroupId:
    """
    TEST 3: test_detector_uses_unique_group_id

    Verifies that StreamingDetector is configured with dynamic group_id
    matching pattern: detector-YYYYMMDD_HHMMSS_microseconds

    EXPECTED FAILURE: Currently uses hardcoded "experiment-detector"
    """

    def test_detector_uses_unique_group_id(self, run_experiment_source):
        """Detector config should use pattern: detector-<timestamp>"""
        detector_id = extract_detector_group_id(run_experiment_source)

        # This test MUST FAIL - pattern should be detector-YYYYMMDD_HHMMSS_microseconds
        pattern = r'^detector-\d{8}_\d{6}_\d{6}$'
        assert re.match(pattern, detector_id or ''), \
            f"FAIL: Detector group_id should match pattern '{pattern}', got: '{detector_id}'"


# ============================================================
# TEST 4: Code checks for unique ID generation at runtime
# ============================================================

class TestCodeUsesDatetimeForId:
    """
    TEST 4: test_code_generates_ids_from_timestamp

    Verifies that the code contains logic to generate unique IDs
    based on datetime timestamp using strftime("%Y%m%d_%H%M%S_%f")

    EXPECTED FAILURE: Currently uses hardcoded strings
    """

    def test_code_generates_ids_from_datetime(self, run_experiment_source):
        """Code should generate IDs using datetime.now().strftime(...)"""
        # Check for the expected timestamp format
        expected_format = r'strftime\s*\(\s*["\']%Y%m%d_%H%M%S_%f["\']\s*\)'

        has_timestamp_format = re.search(expected_format, run_experiment_source)

        # This test MUST FAIL - currently no timestamp generation
        assert has_timestamp_format is not None, \
            "FAIL: Code should use datetime.strftime('%Y%m%d_%H%M%S_%f') to generate unique IDs"


# ============================================================
# TEST 5: Verify experiment_id variable is used for group IDs
# ============================================================

class TestExperimentIdVariableExists:
    """
    TEST 5: test_experiment_id_variable_used

    Verifies that there's an experiment_id variable generated from timestamp
    and used to create group IDs for both FlowConsumer and StreamingDetector.

    EXPECTED FAILURE: Currently no such variable or logic exists
    """

    def test_experiment_id_generated_and_used(self, run_experiment_source):
        """Code should have experiment_id variable based on datetime."""
        # Check for experiment_id generation
        experiment_id_pattern = r'experiment_id\s*=\s*datetime\.now\(\)\.strftime\(["\']%Y%m%d_%H%M%S_%f["\']\)'
        has_id_generation = re.search(experiment_id_pattern, run_experiment_source)

        # This test MUST FAIL - currently no such variable
        assert has_id_generation is not None, \
            "FAIL: Code should generate 'experiment_id = datetime.now().strftime(\"%Y%m%d_%H%M%S_%f\")'"


# ============================================================
# EDGE CASE: Verify format generation is valid
# ============================================================

class TestExperimentIdGenerationPattern:
    """
    Verify that the expected timestamp format can be generated correctly.
    This test passes - it's documentation of the expected format.
    """

    def test_timestamp_format_generation(self):
        """Verify the format can be generated correctly with datetime.strftime."""
        now = datetime.now()
        # Expected format: YYYYMMDD_HHMMSS_microseconds
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")

        # Verify format
        pattern = r'^\d{8}_\d{6}_\d{6}$'
        assert re.match(pattern, timestamp), \
            f"Generated timestamp should match pattern, got: {timestamp}"

        # Example IDs that should be generated
        example_flow_consumer_id = f"flow-consumer-{timestamp}"
        example_detector_id = f"detector-{timestamp}"

        # Both should match their respective patterns
        assert re.match(r'^flow-consumer-\d{8}_\d{6}_\d{6}$', example_flow_consumer_id)
        assert re.match(r'^detector-\d{8}_\d{6}_\d{6}$', example_detector_id)

        # Verify format string used in code
        expected_format = "%Y%m%d_%H%M%S_%f"
        assert len(timestamp) == 22, \
            f"Timestamp should be 22 chars (YYYYMMDD_HHMMSS_microseconds), got {len(timestamp)}"


# ============================================================
# IMPLEMENTATION VERIFICATION
# ============================================================

class TestCurrentImplementationUsesHardcodedIds:
    """
    Verify the current implementation (RED PHASE) uses hardcoded IDs.
    These tests document the current state that needs to be fixed.
    """

    def test_flow_consumer_currently_hardcoded(self, run_experiment_source):
        """Currently FlowConsumer uses hardcoded 'experiment-flow-processor'."""
        flow_consumer_id = extract_flow_consumer_group_id(run_experiment_source)

        # This documents the CURRENT state (RED PHASE)
        assert flow_consumer_id == 'experiment-flow-processor', \
            f"Current implementation uses hardcoded ID, got: {flow_consumer_id}"

    def test_detector_currently_hardcoded(self, run_experiment_source):
        """Currently StreamingDetector uses hardcoded 'experiment-detector'."""
        detector_id = extract_detector_group_id(run_experiment_source)

        # This documents the CURRENT state (RED PHASE)
        assert detector_id == 'experiment-detector', \
            f"Current implementation uses hardcoded ID, got: {detector_id}"

    def test_no_datetime_import_for_ids(self, run_experiment_source):
        """Currently no datetime-based ID generation exists."""
        # Check that there's NO experiment_id variable
        experiment_id_pattern = r'experiment_id\s*=\s*datetime\.now\(\)\.strftime'
        has_id_generation = re.search(experiment_id_pattern, run_experiment_source)

        # This documents that the feature doesn't exist (RED PHASE)
        assert has_id_generation is None, \
            "RED PHASE: experiment_id based on datetime should not exist yet"

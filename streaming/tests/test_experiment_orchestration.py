"""
Unit tests for Experiment Orchestration with Unique Group IDs.

Verifies that unique group IDs are generated per experiment execution:
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
    """Extract the group-id used for FlowConsumer from source.

    Detects both static strings and dynamic f-string patterns.
    Returns 'flow-consumer-{experiment_id}' when the dynamic pattern is found.
    """
    flow_consumer_func = source[source.find('def start_flow_consumer'):source.find('def start_flow_consumer') + 1000]
    # Check for dynamic f-string pattern
    dynamic_pattern = r'--group-id["\']?\s*,\s*f["\']flow-consumer-\{experiment_id\}["\']'
    if re.search(dynamic_pattern, flow_consumer_func):
        return 'flow-consumer-{experiment_id}'
    # Fallback: static string literal
    static_pattern = r'--group-id["\']?\s*,\s*["\']([^"\']+)["\']'
    matches = re.findall(static_pattern, flow_consumer_func)
    return matches[0] if matches else None


def extract_detector_group_id(source):
    """Extract the group_id used for StreamingDetector from source.

    Detects both static strings and dynamic f-string patterns.
    Returns 'detector-{experiment_id}' when the dynamic pattern is found.
    """
    # Check for dynamic f-string pattern
    dynamic_pattern = r'group_id\s*=\s*f["\']detector-\{experiment_id\}["\']'
    if re.search(dynamic_pattern, source):
        return 'detector-{experiment_id}'
    # Fallback: static string literal
    static_pattern = r'group_id\s*=\s*["\']([^"\']+)["\']'
    matches = re.findall(static_pattern, source)
    return matches[0] if matches else None


# ============================================================
# TEST 1: Experiment generates unique ID
# ============================================================

class TestExperimentGeneratesUniqueId:
    """
    TEST 1: test_experiment_generates_unique_id

    Verifies that FlowConsumer group-id uses a dynamic experiment_id,
    not a hardcoded static value.
    """

    def test_experiment_generates_unique_id(self, run_experiment_source):
        """Verify FlowConsumer uses dynamic group-id based on experiment_id."""
        flow_consumer_id = extract_flow_consumer_group_id(run_experiment_source)

        assert flow_consumer_id is not None, "FlowConsumer must have group-id"
        assert flow_consumer_id != 'experiment-flow-processor', \
            f"Group ID should not be hardcoded '{flow_consumer_id}'. Should be dynamic with experiment_id."


# ============================================================
# TEST 2: FlowConsumer uses unique group ID with correct pattern
# ============================================================

class TestFlowConsumerUsesUniqueGroupId:
    """
    TEST 2: test_flow_consumer_uses_unique_group_id

    Verifies that FlowConsumer command includes dynamic group-id
    using f"flow-consumer-{experiment_id}".
    """

    def test_flow_consumer_uses_unique_group_id(self, run_experiment_source):
        """FlowConsumer should use f-string: flow-consumer-{experiment_id}"""
        flow_consumer_id = extract_flow_consumer_group_id(run_experiment_source)

        assert flow_consumer_id == 'flow-consumer-{experiment_id}', \
            f"Group ID should be dynamic f-string 'flow-consumer-{{experiment_id}}', got: '{flow_consumer_id}'"


# ============================================================
# TEST 3: Detector uses unique group ID with correct pattern
# ============================================================

class TestDetectorUsesUniqueGroupId:
    """
    TEST 3: test_detector_uses_unique_group_id

    Verifies that StreamingDetector is configured with dynamic group_id
    using f"detector-{experiment_id}".
    """

    def test_detector_uses_unique_group_id(self, run_experiment_source):
        """Detector config should use f-string: detector-{experiment_id}"""
        detector_id = extract_detector_group_id(run_experiment_source)

        assert detector_id == 'detector-{experiment_id}', \
            f"Detector group_id should be dynamic f-string 'detector-{{experiment_id}}', got: '{detector_id}'"


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



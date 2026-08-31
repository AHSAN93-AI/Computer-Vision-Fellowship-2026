"""
tests.test_state_machine -- Unit tests for ActivityStateMachine

Covers the complete state lifecycle, hysteresis, confirm/end thresholds,
property accessors, and reset behavior.
"""

import time
import pytest
from app.activities.state_machine import ActivityState, ActivityStateMachine


class TestBasicLifecycle:
    """Test the full Idle -> Candidate -> Active -> Ended -> Idle cycle."""

    def test_starts_idle(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=3)
        assert sm.state == ActivityState.IDLE
        assert not sm.is_active

    def test_idle_to_candidate_on_detection(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=3)
        sm.update(True, 0.8)
        assert sm.state == ActivityState.CANDIDATE

    def test_candidate_to_active_after_confirm_frames(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=3)
        for _ in range(3):
            sm.update(True, 0.9)
        assert sm.state == ActivityState.ACTIVE
        assert sm.is_active

    def test_active_to_ended_after_end_frames(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=3)
        # Confirm
        for _ in range(3):
            sm.update(True, 0.9)
        assert sm.state == ActivityState.ACTIVE
        # End
        for _ in range(3):
            sm.update(False, 0.0)
        assert sm.state == ActivityState.ENDED

    def test_ended_to_idle_on_next_update(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=3)
        for _ in range(3):
            sm.update(True, 0.9)
        for _ in range(3):
            sm.update(False, 0.0)
        assert sm.state == ActivityState.ENDED
        # Next update with no detection -> Idle
        sm.update(False, 0.0)
        assert sm.state == ActivityState.IDLE

    def test_ended_to_candidate_on_immediate_redetection(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=3)
        for _ in range(3):
            sm.update(True, 0.9)
        for _ in range(3):
            sm.update(False, 0.0)
        assert sm.state == ActivityState.ENDED
        # Immediately re-detected
        sm.update(True, 0.8)
        assert sm.state == ActivityState.CANDIDATE


class TestCandidateHysteresis:
    """Test that brief detection drops in Candidate state are tolerated."""

    def test_single_drop_tolerated(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=5, end_frames=5)
        sm.update(True, 0.9)   # -> Candidate, detect=1
        sm.update(True, 0.9)   # detect=2
        sm.update(False, 0.0)  # absent=1 (tolerated)
        assert sm.state == ActivityState.CANDIDATE
        sm.update(True, 0.9)   # detect=3, absent reset
        assert sm.state == ActivityState.CANDIDATE

    def test_two_drops_tolerated(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=5, end_frames=5)
        sm.update(True, 0.9)
        sm.update(False, 0.0)  # absent=1
        sm.update(False, 0.0)  # absent=2
        assert sm.state == ActivityState.CANDIDATE

    def test_three_drops_resets_to_idle(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=5, end_frames=5)
        sm.update(True, 0.9)
        sm.update(False, 0.0)  # absent=1
        sm.update(False, 0.0)  # absent=2
        sm.update(False, 0.0)  # absent=3 -> Idle
        assert sm.state == ActivityState.IDLE


class TestConfirmFrames:
    """Test that confirm_frames must be met exactly."""

    def test_not_active_before_threshold(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=10, end_frames=5)
        for _ in range(9):
            sm.update(True, 0.9)
        assert sm.state == ActivityState.CANDIDATE
        assert not sm.is_active

    def test_active_at_threshold(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=10, end_frames=5)
        for _ in range(10):
            sm.update(True, 0.9)
        assert sm.state == ActivityState.ACTIVE
        assert sm.is_active

    def test_single_confirm_frame(self):
        """With confirm_frames=1, becomes Active on the first detection while in Candidate."""
        sm = ActivityStateMachine(activity_type="test", confirm_frames=1, end_frames=5)
        sm.update(True, 0.9)  # Idle -> Candidate (detect_count=1, but first enters Candidate)
        # The code first transitions to CANDIDATE, then checks threshold on subsequent updates
        sm.update(True, 0.9)  # Candidate -> Active (detect_count >= confirm_frames)
        assert sm.state == ActivityState.ACTIVE


class TestEndFrames:
    """Test that end_frames must be met exactly."""

    def test_stays_active_below_threshold(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=5)
        for _ in range(3):
            sm.update(True, 0.9)
        # 4 absent frames (below end_frames=5)
        for _ in range(4):
            sm.update(False, 0.0)
        assert sm.state == ActivityState.ACTIVE

    def test_ends_at_threshold(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=5)
        for _ in range(3):
            sm.update(True, 0.9)
        for _ in range(5):
            sm.update(False, 0.0)
        assert sm.state == ActivityState.ENDED

    def test_detection_resets_absent_count(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=5)
        for _ in range(3):
            sm.update(True, 0.9)
        # 4 absent, then 1 detected, then 4 absent
        for _ in range(4):
            sm.update(False, 0.0)
        sm.update(True, 0.9)  # resets absent count
        for _ in range(4):
            sm.update(False, 0.0)
        assert sm.state == ActivityState.ACTIVE  # still active, only 4 absent


class TestProperties:
    """Test is_active, is_candidate_or_active, confidence, reset."""

    def test_is_candidate_or_active_in_candidate(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=5, end_frames=5)
        sm.update(True, 0.9)
        assert sm.is_candidate_or_active
        assert not sm.is_active

    def test_is_candidate_or_active_when_active(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=2, end_frames=5)
        sm.update(True, 0.9)
        sm.update(True, 0.9)
        assert sm.is_candidate_or_active
        assert sm.is_active

    def test_confidence_tracks_last_value(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=3)
        sm.update(True, 0.5)
        assert sm.confidence == 0.5
        sm.update(True, 0.9)
        assert sm.confidence == 0.9
        sm.update(False, 0.1)
        assert sm.confidence == 0.1

    def test_reset_returns_to_idle(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=2, end_frames=2)
        sm.update(True, 0.9)
        sm.update(True, 0.9)
        assert sm.state == ActivityState.ACTIVE
        sm.reset()
        assert sm.state == ActivityState.IDLE
        assert not sm.is_active

    def test_active_duration_zero_when_idle(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=2, end_frames=2)
        assert sm.active_duration == 0.0

    def test_total_duration_zero_when_idle(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=2, end_frames=2)
        assert sm.total_duration == 0.0


class TestIdleNoOp:
    """Verify Idle state ignores non-detections."""

    def test_idle_stays_idle_on_no_detection(self):
        sm = ActivityStateMachine(activity_type="test", confirm_frames=3, end_frames=3)
        for _ in range(100):
            sm.update(False, 0.0)
        assert sm.state == ActivityState.IDLE

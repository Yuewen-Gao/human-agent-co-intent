"""Tests for event-driven Map Task agent wake-up fingerprints."""
from __future__ import annotations

import unittest

from agent.map_task.event_trigger import follower_trajectory_event_key


class FollowerTrajectoryEventKeyTests(unittest.TestCase):
    def test_ignores_display_metrics_but_detects_a_changed_drawing(self):
        initial = {
            "canvasDataUrl": "data:image/png;base64,first",
            "route_pixel_ratio": 0.10,
            "route_display_metrics": {"image_display_css": {"w": 400, "h": 600}},
        }
        resized_view = {
            **initial,
            "route_display_metrics": {"image_display_css": {"w": 300, "h": 450}},
        }
        changed_drawing = {**initial, "canvasDataUrl": "data:image/png;base64,second"}

        self.assertEqual(
            follower_trajectory_event_key(initial),
            follower_trajectory_event_key(resized_view),
        )
        self.assertNotEqual(
            follower_trajectory_event_key(initial),
            follower_trajectory_event_key(changed_drawing),
        )

    def test_returns_none_without_a_trajectory_payload(self):
        self.assertIsNone(follower_trajectory_event_key({"route_pixel_ratio": 0.10}))

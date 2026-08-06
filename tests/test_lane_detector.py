"""
Unit tests for Autonomous Lane Detection System
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.traditional.hough_detector import HoughLaneDetector
from src.traditional.sliding_window import SlidingWindowDetector
from src.utils.visualization import LaneVisualizer
from src.config import config

class TestLaneDetection(unittest.TestCase):
    
    def setUp(self):
        # Create dummy synthetic road image
        self.height, self.width = 360, 640
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw synthetic white lane lines
        cv2.line(self.image, (100, 360), (280, 200), (255, 255, 255), 10)
        cv2.line(self.image, (540, 360), (360, 200), (255, 255, 255), 10)
        
    def test_hough_detector(self):
        detector = HoughLaneDetector()
        results = detector.detect(self.image)
        
        self.assertIn('left_lane', results)
        self.assertIn('right_lane', results)
        self.assertIn('curvature', results)
        self.assertIn('offset', results)
        self.assertIsInstance(results['curvature'], (int, float))
        self.assertIsInstance(results['offset'], (int, float))
        
    def test_sliding_window(self):
        detector = SlidingWindowDetector()
        results = detector.detect(self.image)
        
        self.assertIn('left_fit', results)
        self.assertIn('right_fit', results)
        self.assertIn('visualization', results)
        self.assertEqual(results['visualization'].shape, self.image.shape)
        
    def test_visualizer(self):
        visualizer = LaneVisualizer()
        left_lane = np.array([100, 360, 280, 200])
        right_lane = np.array([540, 360, 360, 200])
        
        drawn = visualizer.draw_lanes(self.image, left_lane, right_lane)
        self.assertEqual(drawn.shape, self.image.shape)
        
        area = visualizer.draw_lane_area(self.image, left_lane, right_lane)
        self.assertEqual(area.shape, self.image.shape)
        
        text_img = visualizer.draw_curvature_text(self.image, 500.0, 0.1)
        self.assertEqual(text_img.shape, self.image.shape)

if __name__ == '__main__':
    unittest.main()

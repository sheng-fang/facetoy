"""Tests for the camera module."""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from facetoy.io.camera import get_camera_stream, capture_single_frame, get_available_cameras


class TestCameraFunctions:
    """Test cases for camera functions."""
    
    @patch('cv2.VideoCapture')
    def test_get_camera_stream_success(self, mock_video_capture):
        """Test successful camera stream generation."""
        # Mock camera setup
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        # Mock frame data
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, test_frame),
            (True, test_frame),
            (False, None)  # Simulate end of stream
        ]
        
        mock_video_capture.return_value = mock_cap
        
        # Test the generator
        frames = list(get_camera_stream(camera_index=0))
        
        assert len(frames) == 2
        assert all(isinstance(frame, np.ndarray) for frame in frames)
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_get_camera_stream_camera_not_found(self, mock_video_capture):
        """Test camera stream when camera cannot be opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        with pytest.raises(RuntimeError, match="Could not open camera"):
            list(get_camera_stream(camera_index=999))
    
    @patch('cv2.VideoCapture')
    def test_capture_single_frame_success(self, mock_video_capture):
        """Test successful single frame capture."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        mock_video_capture.return_value = mock_cap
        
        frame = capture_single_frame()
        
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_get_available_cameras(self, mock_video_capture):
        """Test getting available cameras."""
        def mock_capture_init(index):
            mock_cap = MagicMock()
            # Simulate cameras available at indices 0 and 1
            mock_cap.isOpened.return_value = index in [0, 1]
            return mock_cap
        
        mock_video_capture.side_effect = mock_capture_init
        
        available = get_available_cameras()
        
        assert available == [0, 1]
        assert mock_video_capture.call_count == 10  # Tests indices 0-9

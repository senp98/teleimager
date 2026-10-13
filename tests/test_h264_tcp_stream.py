import struct
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from teleimager.h264_tcp_stream import (
    CameraStreamRequest,
    H264TcpStreamServer,
    _receive_control_message,
    compose_camera_mosaic,
    compose_camera_output,
    parse_camera_stream_request,
)


def compact_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return bytes((len(encoded),)) + encoded


class H264TcpStreamTest(unittest.TestCase):
    def test_parse_robotree_app_camera_request(self):
        payload = b"\xca\xfe\x01" + struct.pack(
            "<7i",
            1280,
            1080,
            15,
            5_000_000,
            0,
            2,
            12345,
        )
        payload += compact_string("ROBOT_CAMERAS")
        payload += compact_string("192.168.123.20")

        self.assertEqual(
            parse_camera_stream_request(payload),
            CameraStreamRequest(
                width=1280,
                height=1080,
                fps=15,
                bitrate=5_000_000,
                camera="ROBOT_CAMERAS",
                host="192.168.123.20",
                port=12345,
            ),
        )

    def test_parse_camera_request_rejects_invalid_magic(self):
        payload = b"\x00\x00\x01" + bytes(30)
        with self.assertRaisesRegex(ValueError, "magic"):
            parse_camera_stream_request(payload)

    def test_compose_camera_mosaic(self):
        head = np.full((2, 4, 3), (10, 20, 30), dtype=np.uint8)
        left_wrist = np.full((3, 2, 3), (40, 50, 60), dtype=np.uint8)
        right_wrist = np.full((3, 2, 3), (70, 80, 90), dtype=np.uint8)

        mosaic = compose_camera_mosaic(
            head,
            left_wrist,
            right_wrist,
            width=12,
            height=12,
        )

        self.assertEqual(mosaic.shape, (12, 12, 3))
        np.testing.assert_array_equal(mosaic[0, 0], (10, 20, 30))
        np.testing.assert_array_equal(mosaic[10, 1], (40, 50, 60))
        np.testing.assert_array_equal(mosaic[10, 10], (70, 80, 90))

    def test_compose_single_camera_output(self):
        frame = np.full((2, 4, 3), (10, 20, 30), dtype=np.uint8)

        self.assertIs(compose_camera_output((frame,)), frame)

    def test_compose_camera_output_rejects_two_cameras(self):
        frames = (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        )

        with self.assertRaisesRegex(ValueError, "one camera or three cameras"):
            compose_camera_output(frames)

    def test_receive_robotree_app_control_message(self):
        command = b"CLOSE_CAMERA"
        packet = struct.pack("<i", len(command)) + command + struct.pack("<i", 0)
        message = struct.pack(">i", len(packet)) + packet

        self.assertEqual(
            _receive_control_message(MemorySocket(message), threading.Event()),
            ("CLOSE_CAMERA", b""),
        )

    def test_quad_order_and_aspect_ratio(self):
        colors = [(230, 10, 20), (20, 230, 10), (10, 20, 230), (100, 110, 120)]
        frames = tuple(np.full((8, 8, 3), color, dtype=np.uint8) for color in colors)
        output = compose_camera_output(frames, width=80, height=60, layout="stereo_quad")
        self.assertEqual(output.shape, (60, 80, 3))
        for i, color in enumerate(colors):
            x, y = (i % 2) * 40, (i // 2) * 30
            np.testing.assert_array_equal(output[y + 15, x + 20], color)
            self.assertFalse(output[y:y + 30, x:x + 5].any())
            self.assertTrue((output[y:y + 30, x + 5:x + 35] == color).all())

    def test_quad_missing_tiles_are_black_with_nocarema(self):
        frame = np.full((48, 64, 3), 100, dtype=np.uint8)
        with patch("teleimager.h264_tcp_stream.cv2.putText", wraps=cv2.putText) as put_text:
            output = compose_camera_output((frame, None, None, frame), 640, 480, layout="stereo_quad")
        self.assertEqual(put_text.call_count, 2)
        self.assertTrue(all(call.args[1] == "nocarema" for call in put_text.call_args_list))
        self.assertTrue((output[:240, :320] == 100).all())
        self.assertTrue((output[240:, 320:] == 100).all())
        for tile in (output[:240, 320:], output[240:, :320]):
            self.assertFalse(tile[:40].any())
            self.assertGreater(tile[80:160].max(), 0)

    def test_h264_packet_uses_big_endian_length_prefix(self):
        server = H264TcpStreamServer()
        server._request = CameraStreamRequest(
            width=2,
            height=2,
            fps=15,
            bitrate=1_000_000,
            camera="ROBOT_CAMERAS",
            host="127.0.0.1",
            port=12345,
        )
        server._stream_socket = MemorySendSocket()
        server._codec = MemoryCodec()
        server._video_frame = lambda _frame, _request: MemoryFrame()

        self.assertTrue(server.send_frame(np.zeros((2, 2, 3), dtype=np.uint8)))
        self.assertEqual(server._stream_socket.data, b"\x00\x00\x00\x02\x01\x02")


class MemorySocket:
    def __init__(self, data: bytes):
        self.data = data

    def recv(self, size: int) -> bytes:
        chunk = self.data[:size]
        self.data = self.data[size:]
        return chunk


class MemorySendSocket:
    def __init__(self):
        self.data = b""

    def sendall(self, data: bytes):
        self.data += data


class MemoryCodec:
    def encode(self, _frame):
        return [b"\x01\x02"]


class MemoryFrame:
    pts = None


if __name__ == "__main__":
    unittest.main()

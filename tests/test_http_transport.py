from __future__ import annotations

import socket
import time
import unittest
from unittest.mock import patch

from AEGIS.http_transport import _connect_resolved


class _ConnectedSocket:
    def __init__(self) -> None:
        self.timeout_history: list[float] = []
        self.connected_to = None

    def settimeout(self, value: float) -> None:
        self.timeout_history.append(float(value))

    def bind(self, _source_address) -> None:
        return

    def connect(self, socket_address) -> None:
        self.connected_to = socket_address

    def close(self) -> None:
        return


class ResolvedConnectionDeadlineTests(unittest.TestCase):
    def test_successful_multi_address_connect_restores_handshake_budget(self) -> None:
        candidate = _ConnectedSocket()
        addresses = [
            (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 443, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443)),
        ]
        total_budget = 1.0
        deadline = time.monotonic() + total_budget
        with patch("AEGIS.http_transport.socket.socket", return_value=candidate):
            connected = _connect_resolved(addresses, deadline, None)

        self.assertIs(connected, candidate)
        self.assertEqual(candidate.connected_to, addresses[0][4])
        self.assertEqual(len(candidate.timeout_history), 2)
        address_slice, handshake_budget = candidate.timeout_history
        self.assertLessEqual(address_slice, total_budget / 2)
        self.assertGreater(handshake_budget, address_slice * 1.8)
        self.assertLessEqual(handshake_budget, total_budget)


if __name__ == "__main__":
    unittest.main()

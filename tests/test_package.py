import socket

import pytest
import pytest_socket

import ocrbench


def test_version_is_exposed():
    assert ocrbench.__version__ == "0.0.0"


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_unit_tests_have_no_network():
    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)

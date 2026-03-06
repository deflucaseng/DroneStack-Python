import pytest
import cosysairsim as airsim


@pytest.fixture(scope="module")
def client():
    c = airsim.MultirotorClient()
    c.confirmConnection()
    return c


def test_connection(client):
    """Verify that the AirSim server is reachable."""
    # confirmConnection() raises on failure; reaching here means success
    assert client is not None

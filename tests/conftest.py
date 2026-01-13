# tests/conftest.py
import logging
import time
import pytest

from tensorlink.nodes import (
    User,
    Validator,
    Worker,
    UserConfig,
    WorkerConfig,
    ValidatorConfig,
)


PRINT_LEVEL = logging.DEBUG
ON_CHAIN = False
LOCAL = True
UPNP = False


@pytest.fixture(scope="function")
def nodes():
    """
    Create Tensorlink nodes once per test session.
    """

    user = User(
        config=UserConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
        )
    )
    time.sleep(1)

    validator = Validator(
        config=ValidatorConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            endpoint=True,
            endpoint_ip="127.0.0.1",
        )
    )
    time.sleep(1)

    worker = Worker(
        config=WorkerConfig(
            upnp=UPNP, on_chain=ON_CHAIN, local_test=LOCAL, print_level=PRINT_LEVEL
        )
    )
    time.sleep(1)

    yield validator, user, worker

    # Hard cleanup
    user.cleanup()
    worker.cleanup()
    validator.cleanup()
    time.sleep(3)


@pytest.fixture(scope="function")
def connected_nodes(nodes):
    """
    Fully connected local Tensorlink test network.
    """

    validator, user, worker = nodes

    val_key, val_host, val_port = validator.send_request("info", None)

    worker.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)
    user.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)

    return validator, user, worker, (val_key, val_host, val_port)

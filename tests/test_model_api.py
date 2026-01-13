"""
test_model_api.py

This script tests distributed machine learning requests via node API on local nodes.
It simulates an endpoint where model requests, generations, and streamed generations can
be tested on a tiny Hugging Face model.
"""

import requests
import pytest
import time
import json


OFFCHAIN = True
LOCAL = True
UPNP = False

SERVER_URL = "http://127.0.0.1:64747"
MODEL_NAME = "sshleifer/tiny-gpt2"


@pytest.fixture
def requested_model(connected_nodes):
    validator, user, worker, _ = connected_nodes

    payload = {"hf_name": MODEL_NAME, "model_type": "causal"}

    response = requests.post(
        url=f"{SERVER_URL}/request-model",
        json=payload,
        timeout=60,
    )

    assert response.status_code == 200


def test_generate(connected_nodes, requested_model):
    """
    After a model has been successfully requested, try simple generate call
    """
    validator, user, worker, _ = connected_nodes
    time.sleep(5)

    generate_payload = {
        "hf_name": MODEL_NAME,
        "message": "Hi.",
        "max_new_tokens": 10,
        "do_sample": True,
        "num_beams": 2,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/generate",
        json=generate_payload,
        timeout=100,
    )

    assert response.status_code == 200


def test_streaming_generation(connected_nodes, requested_model):
    """
    After a model has been successfully requested, try streamed generate call (token-by-token)
    """
    validator, user, worker, _ = connected_nodes
    time.sleep(5)

    generate_payload = {
        "hf_name": MODEL_NAME,
        "message": "Hi.",
        "max_new_tokens": 10,
        "stream": True,
        "do_sample": False,
        "num_beams": 1,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/generate",
        json=generate_payload,
        stream=True,
        timeout=120,
    )

    assert response.status_code == 200

    full_text = ""
    received_tokens = 0
    done_received = False

    for line in response.iter_lines():
        if not line:
            continue

        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue

        data = decoded[6:]

        if data == "[DONE]":
            done_received = True
            break

        chunk = json.loads(data)
        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content")

        if token:
            received_tokens += 1
            full_text += token

    assert done_received
    assert received_tokens > 0
    assert full_text

"""
Distributed Model on Private Cluster

This example demonstrates how to run a DistributedModel across a private cluster of personal devices using Tensorlink
and PyTorch. Instead of relying on public nodes, you can form a closed network of machines (laptops, desktops, servers)
and distribute model execution across them.

These same devices can also be exposed through Tensorlink’s HTTP API for remote inference, which is covered further
below. In both cases, each participating machine must run the Tensorlink node binary with an appropriate config.json.

To connect devices in a private cluster, you have two options:
    Worker → Validator: Add the validator’s IP:PORT to each worker’s priority_nodes.
    Validator → Workers: Add all worker IP:PORT pairs to the validator’s priority_nodes.

Once nodes are connected, a Python User node can attach to the cluster and execute models using DistributedModel,
or you can submit model and inference requests to the validator endpoint (if enabled).

Worker 1 (config.json)
Runs both a worker and validator and exposes an HTTP endpoint on the local network:
{
  "config": {
    "node": {
      "type": "both",
      "mode": "private",
      "endpoint": true,
      "endpoint_url": "0.0.0.0",
      "endpoint_port": 64747,
      "logging": "INFO"
    },
    "ml": {
      "trusted": false
    }
  }
}

Worker 2 (config.json)
Connects to the validator by specifying its IP:PORT:
{
  "config": {
    "node": {
      "type": "worker",
      "mode": "private",
      "priority_nodes": [
        ["192.168.2.42", 38751]
      ],
      "logging": "INFO"
    },
    "ml": {
      "trusted": false
    }
  }
}
"""

import torch
import logging
from collections import deque
from transformers import AutoTokenizer
from tensorlink.ml import DistributedModel
from tensorlink.nodes import User, UserConfig

MODEL_NAME = "Qwen/Qwen3-8B"

MAX_HISTORY_TURNS = 6
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.4

if __name__ == "__main__":
    # Must explicitly define a User node when connecting to private devices in Python
    user = User(
        UserConfig(priority_nodes=[["192.168.2.42", 38751]])
    )  # Can also connect nodes after init via user.connect_node("ip", port)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DistributedModel(model=MODEL_NAME, training=False, node=user)

    history = deque(maxlen=MAX_HISTORY_TURNS)
    history.append("System: You are a helpful assistant.")
    history.append(f"User: Hello world!")
    prompt = "\n".join(history) + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

from tensorlink.nodes import Worker, WorkerConfig

import torch.cuda as cuda
import subprocess
import logging
import json
import time
import sys
import os


def get_root_dir():
    if getattr(sys, "frozen", False):  # Check if running as an executable
        return os.path.dirname(sys.executable)
    else:  # Running as a Python script
        return os.path.dirname(os.path.abspath(__file__))


def create_env_file(_env_path, _config):
    """
    Create a default .tensorlink.env file at the specified path if it doesn't exist.
    """
    if not os.path.exists(_env_path):
        with open(_env_path, "w") as env_file:
            env_file.write(f"PUBLIC_KEY={_config.get('crypto').get('address')}\n")


def load_config(config_path="config.json"):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            if config.get("config"):
                return config.get("config")
            return config

    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file: {e}")
        return {}


def is_gpu_available(worker_node: Worker):
    try:
        is_loaded = worker_node.send_request("is_loaded", "", timeout=10)
    except Exception as e:
        logging.error(f"Error checking worker node status: {e}")
        is_loaded = False

    if not is_loaded and cuda.is_available():
        # Check if memory is allocated or reserved on the GPU
        # memory_allocated = cuda.memory_allocated()
        # memory_reserved = cuda.memory_reserved()
        # if memory_allocated > 0 or memory_reserved > 0:
        return True
    return False


def start_mining(mining_script, use_sudo=False):
    """
    Start the mining process using the specified script.
    """
    # Convert to absolute path relative to current working directory
    if not os.path.isabs(mining_script):
        mining_script = os.path.abspath(mining_script)

    # Verify script exists
    if not os.path.exists(mining_script):
        raise FileNotFoundError(f"Mining script not found: {mining_script}")

    if use_sudo:
        # Use list form for better security
        return subprocess.Popen(["sudo", mining_script])
    else:
        return subprocess.Popen([mining_script])


def stop_mining(mining_process):
    """
    Stop the mining process if it is running.
    """
    if mining_process and mining_process.poll() is None:  # Check if process is alive
        mining_process.terminate()
        mining_process.wait()


def _confirm_action():
    """
    Prompts the user with a confirmation message before proceeding.
    """
    while True:
        response = (
            input(
                "Trusted mode is enabled. Are you sure you want to proceed? (yes/no, y/n): "
            )
            .strip()
            .lower()
        )
        if response in {"yes", "y"}:
            print("Proceeding with trusted mode.")
            break
        elif response in {"no", "n"}:
            print("Aborting initialization in trusted mode.")
            exit(1)
        else:
            print("Invalid input. Please type 'yes'/'y' or 'no'/'n'.")


def main():
    root_dir = get_root_dir()
    env_path = os.path.join(root_dir, ".tensorlink.env")

    config = load_config(os.path.join(root_dir, "config.json"))
    create_env_file(env_path, config)

    crypto_config = config["crypto"]
    network_config = config["network"]
    ml_config = config["ml"]

    trusted = ml_config.get("trusted", False)
    mode = network_config.get("mode", "private")

    # Network mode presets (keep in sync with validator)
    MODE_PRESETS = {
        "local": dict(local_test=True, upnp=False, on_chain=False),
        "private": dict(local_test=False, upnp=True, on_chain=False),
        "public": dict(local_test=False, upnp=True, on_chain=True),
    }

    try:
        net_flags = MODE_PRESETS[mode]
    except KeyError:
        raise ValueError(f"Unknown network mode: {mode}")

    mining_enabled = crypto_config.get("mining", False)
    mining_script = crypto_config.get("mining-script")
    use_sudo = os.geteuid() == 0

    if trusted:
        _confirm_action()

    worker = Worker(
        config=WorkerConfig(
            **net_flags,
            print_level=logging.INFO,
            priority_nodes=network_config.get("priority_nodes", []),
            seed_validators=crypto_config.get("seed_validators", []),
        ),
        trusted=trusted,
        utilization=True,
    )

    mining_process = None

    try:
        while True:
            if mining_enabled and mining_script:
                if is_gpu_available(worker):
                    if not mining_process or mining_process.poll() is not None:
                        logging.info("Starting mining...")
                        mining_process = start_mining(mining_script, use_sudo)

                        worker.mining_active.value = True
                        time.sleep(2)

                        total_mem = cuda.get_device_properties(0).total_memory
                        reserved = cuda.memory_reserved(0)
                        worker.reserved_memory.value = total_mem - reserved
                else:
                    if mining_process and mining_process.poll() is None:
                        logging.info("Stopping mining...")
                        stop_mining(mining_process)

                        worker.mining_active.value = False
                        worker.reserved_memory.value = 0.0

            time.sleep(5)
            if not worker.node_process.is_alive():
                break

    except KeyboardInterrupt:
        logging.info("Exiting...")
    finally:
        if mining_process:
            stop_mining(mining_process)


if __name__ == "__main__":
    main()

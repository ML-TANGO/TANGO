import json
import os

if "BACKENDAI_CLUSTER_HOST" in os.environ:  # Start mutli-instance setup.
    env = {}
    env["cluster"] = {}
    env["cluster"]["worker"] = []
    env["cluster"]["chief"] = []  # For compatibility
    env["cluster"]["chief"].append("main1:2220")
    for container in os.environ["BACKENDAI_CLUSTER_HOSTS"].split(","):
        if container != "main1":
            env["cluster"]["worker"].append(container + ":2220")
    env["task"] = {}
    if os.environ["BACKENDAI_CLUSTER_ROLE"] == "main":
        env["task"][
            "type"
        ] = "chief"  # For compatibility. Recent TF choose first worker as chief.
        env["task"]["index"] = str(
            int(os.environ["BACKENDAI_CLUSTER_IDX"]) - 1
        )  # Index starts from 0
    else:
        env["task"]["type"] = "worker"
        env["task"]["index"] = str(
            int(os.environ["BACKENDAI_CLUSTER_IDX"]) - 1
        )  # Index starts from 0
    print(json.dumps(env))
else:
    print("")

import yaml
import subprocess
import sys
import shutil
import os
import getpass
from datetime import datetime, timedelta


def current_kst():
    """Get the current time in KST (Korea Standard Time)."""
    UTC_OFFSET = 9
    utc_time = datetime.utcnow()
    kst_time = utc_time + timedelta(hours=UTC_OFFSET)
    return kst_time.strftime("%Y-%m-%d %H:%M:%S KST")


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def write_yaml(data, file_path):
    with open(file_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_inference_with_yaml_config.py <path_to_yaml_config>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    config = read_yaml(config_file_path)

    # Derive the save directory from the model name or path
    model_last_name = config["GENERATE_MODEL_NAME_OR_PATH"].split("/")[-1]
    use_two_servers = config["USE_TWO_SERVERS"]
    save_dir = f"generate/results/{model_last_name}-single-model"

    # Add execution details to the configuration
    config["execution_details"] = {"timestamp_kst": current_kst(), "user": getpass.getuser(), "cwd": os.getcwd()}

    # Save the modified configuration to the output directory
    output_config_path = os.path.join(save_dir, os.path.basename(config_file_path))
    output_dir = os.path.dirname(output_config_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_yaml(config, output_config_path)
    
    # Check if two servers are required
    command = ["vllm/inference_merged.sh", use_two_servers]
    command.extend(
        [
            config["GENERATE_CUDA_DEVICES"],  # 2
            config["GENERATE_MODEL_NAME_OR_PATH"],
            str(config["GENERATE_PORT"]),
            str(config["TENSOR_PARALLEL_SIZE"]), # 5
            config["prompt_key"],
            config["conda_env_name"],  # 7
            config["use_feedback"],
            config["use_iter"],
            str(config["num_iteration"]), # 10
            config["seed_json"],
            config["K"],
            config["dataset_name"],
            config["mode"],
            config["dataset_path"]
        ]
    )
    for i, c in enumerate(command):
        print(i, c)

    # print(command)
    print(f"Num arguments :{len(command)-1}")
    subprocess.run(command)

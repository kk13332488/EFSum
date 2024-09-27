#!/bin/bash

# ----------------- Functions for the single server version -----------------
# function start_server() {
#     echo "Starting server..."
#     sh ${RUN_SERVER} ${CUDA_DEVICES_SINGLE} ${MODEL_PATH_SINGLE} ${TENSOR_SIZE_SINGLE} ${PORT_SINGLE} > ${SERVER_LOG_FILE} 2>&1 &
#     SERVER_PID=$!
#     echo "Server PID: ${SERVER_PID}"

#     echo "Waiting for server to start..."
#     while ! grep "Uvicorn running on" ${SERVER_LOG_FILE}; do
#         sleep 10
#     done
# }
function start_server(){
    echo "Starting server..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_SINGLE python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH_SINGLE \
    --tensor-parallel-size $TENSOR_SIZE_SINGLE \
    --seed 42 \
    --port $PORT_SINGLE > ${SERVER_LOG_FILE} 2>&1 &


    # sh ${RUN_SERVER} ${CUDA_DEVICES_SINGLE} ${MODEL_PATH_SINGLE} ${TENSOR_SIZE_SINGLE} ${PORT_SINGLE} > ${SERVER_LOG_FILE} 2>&1 &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"

    echo "Waiting for server to start..."
    echo "${SERVER_LOG_FILE}"
    while ! grep "Uvicorn running on" ${SERVER_LOG_FILE}; do
        sleep 10
    done
}

function start_run_script_single() {
    echo "Starting the run script..."
    python vllm/inference_with_none_feedback.py  \
    --generation_model_name ${MODEL_PATH_SINGLE} \
    --generation_model_port ${PORT_SINGLE} \
    --do_iterate ${USE_ITER} \
    --iterate_num ${ITER_COUNT} \
    --K ${K} \
    --dataset_name ${DATASET_NAME} \
    --mode ${MODE} \
    --dataset_path ${DATASET_PATH} \
    # \
    # > "${RUN_SCRIPT_LOG_FILE}" &

    RUN_SCRIPT_PID=$!
    # echo "Run script PID: ${RUN_SCRIPT_PID}"

    # echo "Waiting for run script to complete..."
    # while ! grep "Done!" ${RUN_SCRIPT_LOG_FILE}; do
    #     sleep 10
    # done
}

# ----------------- Functions for the two-server version -----------------
# function start_two_servers() {
#     # Feedback Server
#     echo "Starting feedback server..."
#     sh ${RUN_SERVER} ${CUDA_DEVICES_FEEDBACK} ${MODEL_PATH_FEEDBACK} ${TENSOR_SIZE_TWO} ${PORT_FEEDBACK} > ${FEEDBACK_SERVER_LOG_FILE} 2>&1 &
#     FEEDBACK_SERVER_PID=$!
#     echo "Feedback server PID: ${FEEDBACK_SERVER_PID}"

#     # Generate Server
#     echo "Starting generate server..."
#     sh ${RUN_SERVER} ${CUDA_DEVICES_GENERATE} ${MODEL_PATH_GENERATE} ${TENSOR_SIZE_TWO} ${PORT_GENERATE} > ${GENERATE_SERVER_LOG_FILE} 2>&1 &
#     GENERATE_SERVER_PID=$!
#     echo "Generate server PID: ${GENERATE_SERVER_PID}"

#     # Wait for Feedback Server
#     echo "Waiting for feedback server to start..."
#     while ! grep "Uvicorn running on" ${FEEDBACK_SERVER_LOG_FILE}; do
#         sleep 10
#     done

#     # Wait for Generate Server
#     echo "Waiting for generate server to start..."
#     while ! grep "Uvicorn running on" ${GENERATE_SERVER_LOG_FILE}; do
#         sleep 10
#     done

#     echo "Both servers started successfully!"
# }



function handle_sigint() {
    echo -e "\nYou've stopped the main script. The run script process will be terminated."
    kill ${RUN_SCRIPT_PID} || true
    # Ask user if they want to kill the server
    if [ "$USE_TWO_SERVERS" == "yes" ]; then
        kill ${FEEDBACK_SERVER_PID}
        kill ${GENERATE_SERVER_PID}
        echo "Server process terminated."
    else
        kill ${SERVER_PID}
        echo "Server process terminated."
    fi
    exit 0
}


# ----------------- Main Execution -----------------
USE_TWO_SERVERS=$1
RUN_SERVER="vllm/run_server.sh"
FIRST_MODEL_NAME=$(basename $3)
LOG_DIR="vllm/logs"
trap handle_sigint SIGINT
# RUN_SCRIPT_LOG_FILE="${LOG_DIR}/${FIRST_MODEL_NAME}_run_script.log"


# if [ "$#" -ne 1 ]; then
#     echo "Error: Incorrect number of arguments for single-server version."
#     exit 1
# fi
SAVE_DIR="vllm/results/${FIRST_MODEL_NAME}-single-model"
echo "Result will be saved in: ${SAVE_DIR}"
SERVER_LOG_FILE="${LOG_DIR}/${FIRST_MODEL_NAME}_server.log"
CUDA_DEVICES_SINGLE=$2
MODEL_PATH_SINGLE=$3
PORT_SINGLE=$4
TENSOR_SIZE_SINGLE=$5
PROMPT=$6
ENV_NAME=$7
USE_FB=$8
USE_ITER=$9
ITER_COUNT=${10}
SEED_JSON=${11}
K=${12}
DATASET_NAME=${13}
MODE=${14}
DATASET_PATH=${15}

start_server
start_run_script_single
echo "Saved in: ${SAVE_DIR}"
kill ${SERVER_PID}
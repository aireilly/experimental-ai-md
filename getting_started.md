# Getting started

Red Hat AI Inference Server is a container image that optimizes serving and inferencing with LLMs. Using AI Inference Server, you can serve and inference models in a way that boosts their performance while reducing their costs.

# About AI Inference Server

AI Inference Server provides enterprise-grade stability and security, building on upstream, open source software.
AI Inference Server leverages the upstream [vLLM project](https://github.com/vllm-project), which provides state-of-the-art inferencing features.

For example, AI Inference Server uses continuous batching to process requests as they arrive instead of waiting for a full batch to be accumulated. It also uses tensor parallelism to distribute LLM workloads across multiple GPUs.
These features provide reduced latency and higher throughput.

To reduce the cost of inferencing models, AI Inference Server uses paged attention.
LLMs use a mechanism called attention to understand conversations with users.
Normally, attention uses a significant amount of memory, much of which is wasted.
Paged attention addresses this memory wastage by provisioning memory for LLMs similar to the way that virtual memory works for operating systems.
This approach consumes less memory, which lowers costs.

To verify cost savings and performance gains with AI Inference Server, complete the following procedures:

1. [Serving and inferencing with AI Inference Server](#serving-and-inferencing-rhaiis_getting-started)
2. [Validating Red Hat AI Inference Server benefits using key metrics](#validating-benefits-with-key-metrics_getting-started)

# Product and version compatibility

The following table lists the supported product versions for Red Hat AI Inference Server 3.2.1.

**Product and version compatibility**

|     |     |
| --- | --- |
| Product | Supported version |
| Red Hat AI Inference Server | 3.2.1 |
| vLLM core | v0.10.0 |
| LLM Compressor | LLM Compressor v0.6.0 is not included in the Red Hat AI Inference Server 3.2.1 container image. |

# Serving and inferencing with Podman using NVIDIA CUDA AI accelerators

Serve and inference a large language model with Podman and Red Hat AI Inference Server running on NVIDIA CUDA AI accelerators.

* You have installed Podman or Docker.
* You are logged in as a user with sudo access.
* You have access to `registry.redhat.io` and have logged in.
* You have a Hugging Face account and have generated a Hugging Face access token.
* You have access to a Linux server with data center grade NVIDIA AI accelerators installed.
    * For NVIDIA GPUs:
        * [Install NVIDIA drivers](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html)
        * [Install the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
        * If your system has multiple NVIDIA GPUs that use NVswitch, you must have root access to start Fabric Manager

<dl><dt><strong>📌 NOTE</strong></dt><dd>

For more information about supported vLLM quantization schemes for accelerators, see [Supported hardware](https://docs.vllm.ai/en/latest/features/quantization/supported_hardware.html).
</dd></dl>

1. Open a terminal on your server host, and log in to `registry.redhat.io`:

    ```terminal
    $ podman login registry.redhat.io
    ```
2. Pull the relevant the NVIDIA CUDA image by running the following command:

    ```terminal
    $ podman pull registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1
    ```
3. If your system has SELinux enabled, configure SELinux to allow device access:

    ```terminal
    $ sudo setsebool -P container_use_devices 1
    ```
4. Create a volume and mount it into the container. Adjust the container permissions so that the container can use it.

    ```terminal
    $ mkdir -p rhaiis-cache
    ```

    ```terminal
    $ chmod g+rwX rhaiis-cache
    ```
5. Create or append your `HF_TOKEN` Hugging Face token to the `private.env` file.
Source the `private.env` file.

    ```terminal
    $ echo "export HF_TOKEN=<your_HF_token>" > private.env
    ```

    ```terminal
    $ source private.env
    ```
6. Start the AI Inference Server container image.
    1. For NVIDIA CUDA accelerators, if the host system has multiple GPUs and uses NVSwitch, then start NVIDIA Fabric Manager.
    To detect if your system is using NVSwitch, first check if files are present in `/proc/driver/nvidia-nvswitch/devices/`, and then start NVIDIA Fabric Manager.
    Starting NVIDIA Fabric Manager requires root privileges.

        ```terminal
        $ ls /proc/driver/nvidia-nvswitch/devices/
        ```

        **Example output**

        ```terminal
        0000:0c:09.0  0000:0c:0a.0  0000:0c:0b.0  0000:0c:0c.0  0000:0c:0d.0  0000:0c:0e.0
        ```

        ```terminal
        $ systemctl start nvidia-fabricmanager
        ```

        <dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

        NVIDIA Fabric Manager is only required on systems with multiple GPUs that use NVswitch.
        For more information, see [NVIDIA Server Architectures](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html#nvidia-server-architectures).
        </dd></dl>
        1. Check that the Red Hat AI Inference Server container can access NVIDIA GPUs on the host by running the following command:

            ```terminal
            $ podman run --rm -it \
            --security-opt=label=disable \
            --device nvidia.com/gpu=all \
            nvcr.io/nvidia/cuda:12.4.1-base-ubi9 \
            nvidia-smi
            ```

            **Example output**

            ```terminal
            +-----------------------------------------------------------------------------------------+
            | NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
            |-----------------------------------------+------------------------+----------------------+
            | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
            | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
            |                                         |                        |               MIG M. |
            |=========================================+========================+======================|
            |   0  NVIDIA A100-SXM4-80GB          Off |   00000000:08:01.0 Off |                    0 |
            | N/A   32C    P0             64W /  400W |       1MiB /  81920MiB |      0%      Default |
            |                                         |                        |             Disabled |
            +-----------------------------------------+------------------------+----------------------+
            |   1  NVIDIA A100-SXM4-80GB          Off |   00000000:08:02.0 Off |                    0 |
            | N/A   29C    P0             63W /  400W |       1MiB /  81920MiB |      0%      Default |
            |                                         |                        |             Disabled |
            +-----------------------------------------+------------------------+----------------------+

            +-----------------------------------------------------------------------------------------+
            | Processes:                                                                              |
            |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
            |        ID   ID                                                               Usage      |
            |=========================================================================================|
            |  No running processes found                                                             |
            +-----------------------------------------------------------------------------------------+
            ```
        2. Start the container.

            ```terminal
            $ podman run --rm -it \
            --device nvidia.com/gpu=all \
            --security-opt=label=disable \ ①
            --shm-size=4g -p 8000:8000 \ ②
            --userns=keep-id:uid=1001 \ ③
            --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \ ④
            --env "HF_HUB_OFFLINE=0" \
            --env=VLLM_NO_USAGE_STATS=1 \
            -v ./rhaiis-cache:/opt/app-root/src/.cache:Z \ ⑤
            registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1 \
            --model RedHatAI/Llama-3.2-1B-Instruct-FP8 \
            --tensor-parallel-size 2 ⑥
            ```
            1. Required for systems where SELinux is enabled.
            `--security-opt=label=disable` prevents SELinux from relabeling files in the volume mount.
            If you choose not to use this argument, your container might not successfully run.
            2. If you experience an issue with shared memory, increase `--shm-size` to `8GB`.
            3. Maps the host UID to the effective UID of the vLLM process in the container.
            You can also pass `--user=0`, but this less secure than the `--userns` option.
            Setting `--user=0` runs vLLM as root inside the container.
            4. Set and export `HF_TOKEN` with your [Hugging Face API access token](https://huggingface.co/settings/tokens)
            5. Required for systems where SELinux is enabled.
            On Debian or Ubuntu operating systems, or when using Docker without SELinux, the `:Z` suffix is not available.
            6. Set `--tensor-parallel-size` to match the number of GPUs when running the AI Inference Server container on multiple GPUs.
7. In a separate tab in your terminal, make a request to your model with the API.

    ```terminal
    curl -X POST -H "Content-Type: application/json" -d '{
        "prompt": "What is the capital of France?",
        "max_tokens": 50
    }' http://<your_server_ip>:8000/v1/completions | jq
    ```

    **Example output**

    ```json
    {
        "id": "cmpl-b84aeda1d5a4485c9cb9ed4a13072fca",
        "object": "text_completion",
        "created": 1746555421,
        "model": "RedHatAI/Llama-3.2-1B-Instruct-FP8",
        "choices": [
            {
                "index": 0,
                "text": " Paris.\nThe capital of France is Paris.",
                "logprobs": null,
                "finish_reason": "stop",
                "stop_reason": null,
                "prompt_logprobs": null
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 18,
            "completion_tokens": 10,
            "prompt_tokens_details": null
        }
    }
    ```

# Serving and inferencing with Podman using AMD ROCm AI accelerators

Serve and inference a large language model with Podman and Red Hat AI Inference Server running on AMD ROCm AI accelerators.

* You have installed Podman or Docker.
* You are logged in as a user with sudo access.
* You have access to `registry.redhat.io` and have logged in.
* You have a Hugging Face account and have generated a Hugging Face access token.
* You have access to a Linux server with data center grade AMD ROCm AI accelerators installed.
    * For AMD GPUs:
        * [Install ROCm software](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/)
        * [Verify that you can run ROCm containers](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/how-to/docker.html)

<dl><dt><strong>📌 NOTE</strong></dt><dd>

For more information about supported vLLM quantization schemes for accelerators, see [Supported hardware](https://docs.vllm.ai/en/latest/features/quantization/supported_hardware.html).
</dd></dl>

1. Open a terminal on your server host, and log in to `registry.redhat.io`:

    ```terminal
    $ podman login registry.redhat.io
    ```
2. Pull the AMD ROCm image by running the following command:

    ```terminal
    $ podman pull registry.redhat.io/rhaiis/vllm-rocm-rhel9:3.2.1
    ```
3. If your system has SELinux enabled, configure SELinux to allow device access:

    ```terminal
    $ sudo setsebool -P container_use_devices 1
    ```
4. Create a volume and mount it into the container. Adjust the container permissions so that the container can use it.

    ```terminal
    $ mkdir -p rhaiis-cache
    ```

    ```terminal
    $ chmod g+rwX rhaiis-cache
    ```
5. Create or append your `HF_TOKEN` Hugging Face token to the `private.env` file.
Source the `private.env` file.

    ```terminal
    $ echo "export HF_TOKEN=<your_HF_token>" > private.env
    ```

    ```terminal
    $ source private.env
    ```
6. Start the AI Inference Server container image.
    1. For AMD ROCm accelerators:
        1. Use `amd-smi static -a` to verify that the container can access the host system GPUs:

            ```terminal
            $ podman run -ti --rm --pull=newer \
            --security-opt=label=disable \
            --device=/dev/kfd --device=/dev/dri \
            --group-add keep-groups \ ①
            --entrypoint="" \
            registry.redhat.io/rhaiis/vllm-rocm-rhel9:3.2.1 \
            amd-smi static -a
            ```
            1. You must belong to both the video and render groups on AMD systems to use the GPUs.
            To access GPUs, you must pass the `--group-add=keep-groups` supplementary groups option into the container.
        2. Start the container:

            ```terminal
            podman run --rm -it \
            --device /dev/kfd --device /dev/dri \
            --security-opt=label=disable \ ①
            --group-add keep-groups \
            --shm-size=4GB -p 8000:8000 \ ②
            --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
            --env "HF_HUB_OFFLINE=0" \
            --env=VLLM_NO_USAGE_STATS=1 \
            -v ./rhaiis-cache:/opt/app-root/src/.cache \
            registry.redhat.io/rhaiis/vllm-rocm-rhel9:3.2.1 \
            --model RedHatAI/Llama-3.2-1B-Instruct-FP8 \
            --tensor-parallel-size 2 ③
            ```
            1. `--security-opt=label=disable` prevents SELinux from relabeling files in the volume mount. If you choose not to use this argument, your container might not successfully run.
            2. If you experience an issue with shared memory, increase `--shm-size` to `8GB`.
            3. Set `--tensor-parallel-size` to match the number of GPUs when running the AI Inference Server container on multiple GPUs.
7. In a separate tab in your terminal, make a request to the model with the API.

    ```terminal
    curl -X POST -H "Content-Type: application/json" -d '{
        "prompt": "What is the capital of France?",
        "max_tokens": 50
    }' http://<your_server_ip>:8000/v1/completions | jq
    ```

    **Example output**

    ```json
    {
        "id": "cmpl-b84aeda1d5a4485c9cb9ed4a13072fca",
        "object": "text_completion",
        "created": 1746555421,
        "model": "RedHatAI/Llama-3.2-1B-Instruct-FP8",
        "choices": [
            {
                "index": 0,
                "text": " Paris.\nThe capital of France is Paris.",
                "logprobs": null,
                "finish_reason": "stop",
                "stop_reason": null,
                "prompt_logprobs": null
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 18,
            "completion_tokens": 10,
            "prompt_tokens_details": null
        }
    }
    ```

# Serving and inferencing language models with Podman using Google TPU AI accelerators

Serve and inference a large language model with Podman or Docker and Red Hat AI Inference Server in a Google cloud VM that has Google TPU AI accelerators available.

* You have access to a Google cloud TPU VM with Google TPU AI accelerators configured.
For more information, see:
    * [Set up the Cloud TPU environment](https://cloud.google.com/tpu/docs/setup-gcp-account)
    * [vLLM inference on v6e TPUs](https://cloud.google.com/tpu/docs/tutorials/LLM/vllm-inference-v6e)
* You have installed Podman or Docker.
* You are logged in as a user with sudo access.
* You have access to the `registry.redhat.io` image registry and have logged in.
* You have a Hugging Face account and have generated a Hugging Face access token.

<dl><dt><strong>📌 NOTE</strong></dt><dd>

For more information about supported vLLM quantization schemes for accelerators, see [Supported hardware](https://docs.vllm.ai/en/latest/features/quantization/supported_hardware.html).
</dd></dl>

1. Open a terminal on your TPU server host, and log in to `registry.redhat.io`:

    ```terminal
    $ podman login registry.redhat.io
    ```
2. Pull the Red Hat AI Inference Server image by running the following command:

    ```terminal
    $ podman pull registry.redhat.io/rhaiis/vllm-tpu-rhel9:3.2.1
    ```
3. Optional: Verify that the TPUs are available in the host.
    1. Open a shell prompt in the Red Hat AI Inference Server container. Run the following command:

        ```terminal
        $ podman run -it --net=host --privileged -e PJRT_DEVICE=TPU --rm --entrypoint /bin/bash registry.redhat.io/rhaiis/vllm-tpu-rhel9:3.2.1
        ```
    2. Verify system TPU access and basic operations by running the following Python code in the container shell prompt:

        ```terminal
        $ python3 -c "
        import torch
        import torch_xla
        try:
            device = torch_xla.device()
            print(f'')
            print(f'XLA device available: {device}')
            x = torch.randn(3, 3).to(device)
            y = torch.randn(3, 3).to(device)
            z = torch.matmul(x, y)
            import torch_xla.core.xla_model as xm
            torch_xla.sync()
            print(f'Matrix multiplication successful')
            print(f'Result tensor shape: {z.shape}')
            print(f'Result tensor device: {z.device}')
            print(f'Result tensor: {z.data}')
            print('TPU is operational.')
        except Exception as e:
            print(f'TPU test failed: {e}')
            print('Try restarting the container to clear TPU locks')
        "
        ```

        **Example output**

        ```terminal
        XLA device available: xla:0
        Matrix multiplication successful
        Result tensor shape: torch.Size([3, 3])
        Result tensor device: xla:0
        Result tensor: tensor([[-1.8161,  1.6359, -3.1301],
                [-1.2205,  0.8985, -1.4422],
                [ 0.0588,  0.7693, -1.5683]], device='xla:0')
        TPU is operational.
        ```
    3. Exit the shell prompt.

        ```terminal
        $ exit
        ```
4. Create a volume and mount it into the container. Adjust the container permissions so that the container can use it.

    ```terminal
    $ mkdir ./.cache/rhaiis
    ```

    ```terminal
    $ chmod g+rwX ./.cache/rhaiis
    ```
5. Add the `HF_TOKEN` Hugging Face token to the `private.env` file.

    ```terminal
    $ echo "export HF_TOKEN=<huggingface_token>" > private.env
    ```
6. Append the `HF_HOME` variable to the `private.env` file.

    ```terminal
    $ echo "export HF_HOME=./.cache/rhaiis" >> private.env
    ```

    Source the `private.env` file.

    ```terminal
    $ source private.env
    ```
7. Start the AI Inference Server container image:

    ```terminal
    podman run --rm -it \
      --name vllm-tpu \
      --network=host \
      --privileged \
      --shm-size=4g \
      --device=/dev/vfio/vfio \
      --device=/dev/vfio/0 \
      -e PJRT_DEVICE=TPU \
      -e HF_HUB_OFFLINE=0 \
      -v ./.cache/rhaiis:/opt/app-root/src/.cache \
      registry.redhat.io/rhaiis/vllm-tpu-rhel9:3.2.1 \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --tensor-parallel-size 1 \ ①
      --max-model-len=256 \
      --host=0.0.0.0 \
      --port=8000
    ```
    1. Set `--tensor-parallel-size` to match the number of TPUs.

Check that the AI Inference Server server is up. Open a separate tab in your terminal, and make a model request with the API:

```terminal
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Briefly, what colour is the wind?"}
    ],
    "max_tokens": 50
  }' | jq
```

**Example output**

```json
{
  "id": "chatcmpl-13a9d6a04fd245409eb601688d6144c1",
  "object": "chat.completion",
  "created": 1755268559,
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The wind is typically associated with the color white or grey, as it can carry dust, sand, or other particles. However, it is not a color in the traditional sense.",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning_content": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 38,
    "total_tokens": 75,
    "completion_tokens": 37,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "kv_transfer_params": null
}
```

* [Introduction to Cloud TPU](https://cloud.google.com/tpu/docs/intro-to-tpu)

# Validating Red Hat AI Inference Server benefits using key metrics

Use the following metrics to evaluate the performance of the LLM model being served with AI Inference Server:

* ***Time to first token (TTFT)***: The time from when a request is sent to when the first token of the response is received.
* ***Time per output token (TPOT)***: The average time it takes to generate each token after the first one.
* ***Latency***: The total time required to generate the full response.
* ***Throughput***: The total number of output tokens the model can produce at the same time across all users and requests.

Complete the procedure below to run a benchmark test that shows how AI Inference Server, and other inference servers, perform according to these metrics.

**Prerequisites**

* AI Inference Server container image
* GitHub account
* Python 3.9 or higher

**Procedure**

1. On your host system, start an AI Inference Server container and serve a model.

    ```terminal
    $ podman run --rm -it --device nvidia.com/gpu=all \
    --shm-size=4GB -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --env "HF_HUB_OFFLINE=0" \
    -v ./rhaiis-cache:/opt/app-root/src/.cache \
    --security-opt=label=disable \
    registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1 \
    --model RedHatAI/Llama-3.2-1B-Instruct-FP8
    ```
2. In a separate terminal tab, install the benchmark tool dependencies.

    ```terminal
    $ pip install vllm pandas datasets
    ```
3. Clone the [vLLM Git repository](https://github.com/vllm-project/vllm):

    ```terminal
    $ git clone https://github.com/vllm-project/vllm.git
    ```
4. Run the `./vllm/benchmarks/benchmark_serving.py` script.

    ```terminal
    $ python vllm/benchmarks/benchmark_serving.py --backend vllm --model RedHatAI/Llama-3.2-1B-Instruct-FP8 --num-prompts 100 --dataset-name random  --random-input 1024 --random-output 512 --port 8000
    ```

The results show how AI Inference Server performs according to key server metrics:

```text
============ Serving Benchmark Result ============
Successful requests:                    100
Benchmark duration (s):                 4.61
Total input tokens:                     102300
Total generated tokens:                 40493
Request throughput (req/s):             21.67
Output token throughput (tok/s):        8775.85
Total Token throughput (tok/s):         30946.83
---------------Time to First Token----------------
Mean TTFT (ms):                         193.61
Median TTFT (ms):                       193.82
P99 TTFT (ms):                          303.90
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                         9.06
Median TPOT (ms):                       8.57
P99 TPOT (ms):                          13.57
---------------Inter-token Latency----------------
Mean ITL (ms):                          8.54
Median ITL (ms):                        8.49
P99 ITL (ms):                           13.14
==================================================
```

Try changing the parameters of this benchmark and running it again. Notice how `vllm` as a backend compares to other options. Throughput should be consistently higher, while latency should be lower.

* Other options for `--backend` are: `tgi`, `lmdeploy`, `deepspeed-mii`, `openai`, and `openai-chat`
* Other options for `--dataset-name` are: `sharegpt`, `burstgpt`, `sonnet`, `random`, `hf`

**Additional resources**

* [vLLM documentation](https://docs.vllm.ai/en/latest/)
* [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices), by Mosaic AI Research, which explains metrics such as throughput and latency

# Troubleshooting

The following troubleshooting information for Red Hat AI Inference Server 3.2.1 describes common problems related to model loading, memory, model response quality, networking, and GPU drivers.
Where available, workarounds for common issues are described.

Most common issues in vLLM relate to installation, model loading, memory management, and GPU communication.
Most problems can be resolved by using a correctly configured environment, ensuring compatible hardware and software versions, and following the recommended configuration practices.

<dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

For persistent issues, export `VLLM_LOGGING_LEVEL=DEBUG` to enable debug logging and then check the logs.

```terminal
$ export VLLM_LOGGING_LEVEL=DEBUG
```
</dd></dl>

## Model loading errors

* When you run the Red Hat AI Inference Server container image without specifying a user namespace, an unrecognized model error is returned.

    ```terminal
    podman run --rm -it \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    --shm-size=4GB -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --env "HF_HUB_OFFLINE=0" \
    --env=VLLM_NO_USAGE_STATS=1 \
    -v ./rhaiis-cache:/opt/app-root/src/.cache \
    registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1 \
    --model RedHatAI/Llama-3.2-1B-Instruct-FP8
    ```

    **Example output**

    ```terminal
    ValueError: Unrecognized model in RedHatAI/Llama-3.2-1B-Instruct-FP8. Should have a model_type key in its config.json
    ```

    To resolve this error, pass `--userns=keep-id:uid=1001` as a Podman parameter to ensure that the container runs with the root user.
* Sometimes when Red Hat AI Inference Server downloads the model, the download fails or gets stuck.
To prevent the model download from hanging, first download the model using the `huggingface-cli`.
For example:

    ```terminal
    $ huggingface-cli download <MODEL_ID> --local-dir <DOWNLOAD_PATH>
    ```

    When serving the model, pass the local model path to vLLM to prevent the model from being downloaded again.
* When Red Hat AI Inference Server loads a model from disk, the process sometimes hangs.
Large models consume memory, and if memory runs low, the system slows down as it swaps data between RAM and disk.
Slow network file system speeds or a lack of available memory can trigger excessive swapping.
This can happen in clusters where file systems are shared between cluster nodes.

    Where possible, store the model in a local disk to prevent slow down during model loading.
    Ensure that the system has sufficient CPU memory available.

    Ensure that your system has enough CPU capacity to handle the model.
* Sometimes, Red Hat AI Inference Server fails to inspect the model.
Errors are reported in the log.
For example:

    ```terminal
    #...
      File "vllm/model_executor/models/registry.py", line xxx, in \_raise_for_unsupported
        raise ValueError(
    ValueError: Model architectures [''] failed to be inspected. Please check the logs for more details.
    ```

    The error occurs when vLLM fails to import the model file, which is usually related to missing dependencies or outdated binaries in the vLLM build.
* Some model architectures are not supported.
Refer to the list of [Validated models](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.2/html-single/validated_models/index).
For example, the following errors indicate that the model you are trying to use is not supported:

    ```terminal
    Traceback (most recent call last):
    #...
      File "vllm/model_executor/models/registry.py", line xxx, in inspect_model_cls
        for arch in architectures:
    TypeError: 'NoneType' object is not iterable
    ```

    ```terminal
    #...
      File "vllm/model_executor/models/registry.py", line xxx, in \_raise_for_unsupported
        raise ValueError(
    ValueError: Model architectures [''] are not supported for now. Supported architectures:
    #...
    ```

    <dl><dt><strong>📌 NOTE</strong></dt><dd>

    Some architectures such as `DeepSeekV2VL` require the architecture to be explicitly specified using the `--hf_overrides` flag, for example:

    ```terminal
    --hf_overrides '{\"architectures\": [\"DeepseekVLV2ForCausalLM\"]}
    ```
    </dd></dl>
* Sometimes a runtime error occurs for certain hardware when you load 8-bit floating point (FP8) models.
FP8 requires GPU hardware acceleration.
Errors occur when you load FP8 models like `deepseek-r1` or models tagged with the `F8_E4M3` tensor type.
For example:

    ```terminal
    triton.compiler.errors.CompilationError: at 1:0:
    def \_per_token_group_quant_fp8(
    \^
    ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
    [rank0]:[W502 11:12:56.323757996 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
    ```

    <dl><dt><strong>📌 NOTE</strong></dt><dd>

    Review [Getting started](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.2/html-single/getting_started/index) to ensure your specific accelerator is supported.
    Accelerators that are currently supported for FP8 models include:

    * [NVIDIA CUDA T4, A100, L4, L40S, H100, and H200 GPUs](https://developer.nvidia.com/cuda-gpus)
    * [AMD ROCm MI300X GPUs](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
    </dd></dl>
* Sometimes when serving a model a runtime error occurs that is related to the host system.
For example, you might see errors in the log like this:

    ```terminal
    INFO 05-07 19:15:17 [config.py:1901] Chunked prefill is enabled with max_num_batched_tokens=2048.
    OMP: Error #179: Function Can't open SHM failed:
    OMP: System error #0: Success
    Traceback (most recent call last):
      File "/opt/app-root/bin/vllm", line 8, in <module>
        sys.exit(main())
    ..........................    raise RuntimeError("Engine core initialization failed. "
    RuntimeError: Engine core initialization failed. See root cause above.
    ```

    You can work around this issue by passing the `--shm-size=2g` argument when starting `vllm`.

## Memory optimization

* If the model is too large to run with a single GPU, you will get out-of-memory (OOM) errors.
Use memory optimization options such as quantization, tensor parallelism, or reduced precision to reduce the memory consumption.
For more information, see [Conserving memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory.html).

## Generated model response quality

* In some scenarios, the quality of the generated model responses might deteriorate after an update.

    Default sampling parameters source have been updated in newer versions.
    For vLLM version 0.8.4 and higher, the default sampling parameters come from the `generation_config.json` file that is provided by the model creator.
    In most cases, this should lead to higher quality responses, because the model creator is likely to know which sampling parameters are best for their model.
    However, in some cases the defaults provided by the model creator can lead to degraded performance.

    If you experience this problem, try serving the model with the old defaults by using the `--generation-config vllm` server argument.

    <dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

    If applying the `--generation-config vllm` server argument improves the model output, continue to use the vLLM defaults and petition the model creator on [Hugging Face](https://huggingface.co) to update their default `generation_config.json` so that it produces better quality generations.
    </dd></dl>

## CUDA accelerator errors

* You might experience a `self.graph.replay()` error when running a model using CUDA accelerators.

    If vLLM crashes and the error trace captures the error somewhere around the `self.graph.replay()` method in the `vllm/worker/model_runner.py` module, this is most likely a CUDA error that occurs inside the `CUDAGraph` class.

    To identify the particular CUDA operation that causes the error, add the `--enforce-eager` server argument to the `vllm` command line to disable `CUDAGraph` optimization and isolate the problematic CUDA operation.
* You might experience accelerator and CPU communication problems that are caused by incorrect hardware or driver settings.

    NVIDIA Fabric Manager is required for multi-GPU systems for some types of NVIDIA GPUs.
    The `nvidia-fabricmanager` package and associated systemd service might not be installed or the package might not be running.

    Run the [diagnostic Python script](https://docs.vllm.ai/en/latest/usage/troubleshooting.html#incorrect-hardwaredriver) to check whether the NVIDIA Collective Communications Library (NCCL) and Gloo library components are communicating correctly.

    On an NVIDIA system, check the fabric manager status by running the following command:

    ```terminal
    $ systemctl status nvidia-fabricmanager
    ```

    On successfully configured systems, the service should be active and running with no errors.
* Running vLLM with tensor parallelism enabled and setting `--tensor-parallel-size` to be greater than 1 on NVIDIA Multi-Instance GPU (MIG) hardware causes an `AssertionError` during the initial model loading or shape checking phase.
This typically occurs as one of the first errors when starting vLLM.

## Networking errors

* You might experience network errors with complicated network configurations.

    To troubleshoot network issues, search the logs for DEBUG statements where an incorrect IP address is listed, for example:

    ```terminal
    DEBUG 06-10 21:32:17 parallel_state.py:88] world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://<incorrect_ip_address>:54641 backend=nccl
    ```

    To correct the issue, set the correct IP address with the `VLLM_HOST_IP` environment variable, for example:
    ```terminal
    $ export VLLM_HOST_IP=<correct_ip_address>
    ```

    Specify the network interface that is tied to the IP address for NCCL and Gloo:

    ```terminal
    $ export NCCL_SOCKET_IFNAME=<your_network_interface>
    ```

    ```terminal
    $ export GLOO_SOCKET_IFNAME=<your_network_interface>
    ```

## Python multiprocessing errors

* You might experience Python multiprocessing warnings or runtime errors.
This can be caused by code that is not properly structured for Python multiprocessing.
The following is an example console warning:

    ```terminal
    WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
        initialized. We must use the `spawn` multiprocessing start method. Setting
        VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
        https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#python-multiprocessing
        for more information.
    ```

    The following is an example Python runtime error:

    ```terminal
    RuntimeError:
            An attempt has been made to start a new process before the
            current process has finished its bootstrapping phase.

            This probably means that you are not using fork to start your
            child processes and you have forgotten to use the proper idiom
            in the main module:

                if __name__ = "__main__":
                    freeze_support()
                    ...

            The "freeze_support()" line can be omitted if the program
            is not going to be frozen to produce an executable.

            To fix this issue, refer to the "Safe importing of main module"
            section in https://docs.python.org/3/library/multiprocessing.html
    ```

    To resolve the runtime error, update your Python code to guard the usage of `vllm` behind an `+if__name__ = "__main__":+` block, for example:

    ```python
    if __name__ = "__main__":
        import vllm

        llm = vllm.LLM(...)
    ```

## GPU driver or device pass-through issues

* When you run the Red Hat AI Inference Server container image, sometimes it is unclear whether device pass-through errors are being caused by GPU drivers or tools such as the NVIDIA Container Toolkit.

    * Check that the NVIDIA Container toolkit that is installed on the host machine can see the host GPUs:

        ```terminal
        $ nvidia-ctk cdi list
        ```

        **Example output**

        ```terminal
        #...
        nvidia.com/gpu=GPU-0fe9bb20-207e-90bf-71a7-677e4627d9a1
        nvidia.com/gpu=GPU-10eff114-f824-a804-e7b7-e07e3f8ebc26
        nvidia.com/gpu=GPU-39af96b4-f115-9b6d-5be9-68af3abd0e52
        nvidia.com/gpu=GPU-3a711e90-a1c5-3d32-a2cd-0abeaa3df073
        nvidia.com/gpu=GPU-6f5f6d46-3fc1-8266-5baf-582a4de11937
        nvidia.com/gpu=GPU-da30e69a-7ba3-dc81-8a8b-e9b3c30aa593
        nvidia.com/gpu=GPU-dc3c1c36-841b-bb2e-4481-381f614e6667
        nvidia.com/gpu=GPU-e85ffe36-1642-47c2-644e-76f8a0f02ba7
        nvidia.com/gpu=all
        ```
    * Ensure that the NVIDIA accelerator configuration has been created on the host machine:

        ```terminal
        $ sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
        ```
    * Check that the Red Hat AI Inference Server container can access NVIDIA GPUs on the host by running the following command:

        ```terminal
        $ podman run --rm -it --security-opt=label=disable --device nvidia.com/gpu=all nvcr.io/nvidia/cuda:12.4.1-base-ubi9 nvidia-smi
        ```

        **Example output**

        ```terminal
        +-----------------------------------------------------------------------------------------+
        | NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
        |-----------------------------------------+------------------------+----------------------+
        | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
        |                                         |                        |               MIG M. |
        |=========================================+========================+======================|
        |   0  NVIDIA A100-SXM4-80GB          Off |   00000000:08:01.0 Off |                    0 |
        | N/A   32C    P0             64W /  400W |       1MiB /  81920MiB |      0%      Default |
        |                                         |                        |             Disabled |
        +-----------------------------------------+------------------------+----------------------+
        |   1  NVIDIA A100-SXM4-80GB          Off |   00000000:08:02.0 Off |                    0 |
        | N/A   29C    P0             63W /  400W |       1MiB /  81920MiB |      0%      Default |
        |                                         |                        |             Disabled |
        +-----------------------------------------+------------------------+----------------------+

        +-----------------------------------------------------------------------------------------+
        | Processes:                                                                              |
        |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
        |        ID   ID                                                               Usage      |
        |=========================================================================================|
        |  No running processes found                                                             |
        +-----------------------------------------------------------------------------------------+
        ```

# Gathering system information with the vLLM collect environment script

Use the `vllm collect-env` command that you run from the Red Hat AI Inference Server container to gather system information for troubleshooting AI Inference Server deployments.
This script collects system details, hardware configurations, and dependency information that can help diagnose deployment problems and model inference serving issues.

* You have installed Podman or Docker.
* You are logged in as a user with sudo access.
* You have access to a Linux server with data center grade AI accelerators installed.
* You have pulled and successfully deployed the Red Hat AI Inference Server container.

1. Open a terminal and log in to `registry.redhat.io`:

    ```terminal
    $ podman login registry.redhat.io
    ```
2. Pull the specific Red Hat AI Inference Server container image for the AI accelerator that is installed.
For example, to pull the Red Hat AI Inference Server container for Google cloud TPUs, run the following command:

    ```terminal
    $ podman pull registry.redhat.io/rhaiis/vllm-tpu-rhel9:3.2.1
    ```
3. Run the collect environment script in the container:

    ```terminal
    $ podman run --rm -it \
      --name vllm-tpu \
      --network=host \
      --privileged \
      --device=/dev/vfio/vfio \
      --device=/dev/vfio/0 \
      -e PJRT_DEVICE=TPU \
      -e HF_HUB_OFFLINE=0 \
      -v ./.cache/rhaiis:/opt/app-root/src/.cache:Z \
      --entrypoint vllm collect-env \
      registry.redhat.io/rhaiis/vllm-tpu-rhel9:3.2.1
    ```

The `vllm collect-env` command output details environment information including the following:

* System hardware details
* Operating system details
* Python environment and dependencies
* GPU/TPU accelerator information

Review the output for any warnings or errors that might indicate configuration issues.
Include the `collect-env` output for your system when reporting problems to Red Hat Support.

An example Google Cloud TPU report is provided below:

```terminal
==============================
        System Info
==============================
OS                           : Red Hat Enterprise Linux 9.6 (Plow) (x86_64)
GCC version                  : (GCC) 11.5.0 20240719 (Red Hat 11.5.0-5)
Clang version                : Could not collect
CMake version                : version 4.1.0
Libc version                 : glibc-2.34

==============================
       PyTorch Info
==============================
PyTorch version              : 2.9.0.dev20250716
Is debug build               : False
CUDA used to build PyTorch   : None
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.9 (main, Jun 20 2025, 00:00:00) [GCC 11.5.0 20240719 (Red Hat 11.5.0-5)] (64-bit runtime)
Python platform              : Linux-6.8.0-1015-gcp-x86_64-with-glibc2.34

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : False
CUDA runtime version         : No CUDA
CUDA_MODULE_LOADING set to   : N/A
GPU models and configuration : No CUDA
Nvidia driver version        : No CUDA
cuDNN version                : No CUDA
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : True

==============================
          CPU Info
==============================
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        52 bits physical, 57 bits virtual
Byte Order:                           Little Endian
CPU(s):                               44
On-line CPU(s) list:                  0-43
Vendor ID:                            AuthenticAMD
Model name:                           AMD EPYC 9B14
CPU family:                           25
Model:                                17
Thread(s) per core:                   2
Core(s) per socket:                   22
Socket(s):                            1
Stepping:                             1
BogoMIPS:                             5200.00
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb stibp vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr wbnoinvd arat avx512vbmi umip avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm
Hypervisor vendor:                    KVM
Virtualization type:                  full
L1d cache:                            704 KiB (22 instances)
L1i cache:                            704 KiB (22 instances)
L2 cache:                             22 MiB (22 instances)
L3 cache:                             96 MiB (3 instances)
NUMA node(s):                         1
NUMA node0 CPU(s):                    0-43
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Mitigation; Safe RET
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Retpolines; IBPB conditional; IBRS_FW; STIBP always-on; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected

==============================
Versions of relevant libraries
==============================
[pip3] numpy==1.26.4
[pip3] pyzmq==27.0.1
[pip3] torch==2.9.0.dev20250716
[pip3] torch-xla==2.9.0.dev20250716
[pip3] torchvision==0.24.0.dev20250716
[pip3] transformers==4.55.2
[pip3] triton==3.3.1
[conda] Could not collect

==============================
         vLLM Info
==============================
ROCM Version                 : Could not collect
Neuron SDK Version           : N/A
vLLM Version                 : 0.10.0+rhai1
vLLM Build Flags:
  CUDA Archs: Not Set; ROCm: Disabled; Neuron: Disabled
GPU Topology:
  Could not collect

==============================
     Environment Variables
==============================
VLLM_USE_V1=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_NO_USAGE_STATS=1
NCCL_CUMEM_ENABLE=0
PYTORCH_NVML_BASED_CUDA_CHECK=1
TORCHINDUCTOR_COMPILE_THREADS=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_default
```

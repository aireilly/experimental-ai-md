# vLLM server arguments

Red Hat AI Inference Server provides an OpenAI-compatible API server for inference serving.
You can control the behavior of the server with arguments.

This document begins with a list of the most important server arguments that you use with the `vllm serve` command.
A complete list of `vllm serve` arguments, environment variables, server metrics are also provided.

# Key vLLM server arguments

There are 4 key arguments that you use to configure AI Inference Server to run on your hardware:

1. [`--tensor-parallel-size`](#tensor-parallel-size): distributes your model across your host GPUs.
2. [`--gpu-memory-utilization`](#gpu-memory-utilization): adjusts accelerator memory utilization for model weights, activations, and KV cache. Measured as a fraction from 0.0 to 1.0 that defaults to 0.9. For example, you can set this value to 0.8 to limit GPU memory consumption by AI Inference Server to 80%. Use the largest value that is stable for your deployment to maximize throughput.
3. [`--max-model-len`](#max-model-len): limits the maximum context length of the model, measured in tokens. Set this to prevent problems with memory if the model’s default context length is too long.
4. [`--max-num-batched-tokens`](#max-num-batched-tokens): limits the maximum batch size of tokens to process per step, measured in tokens. Increasing this improves throughput but can affect output token latency.

For example, to run the Red Hat AI Inference Server container and serve a model with vLLM, run the following, changing server arguments as required:

```terminal
$ podman run --rm -it \
--device nvidia.com/gpu=all \
--security-opt=label=disable \
--shm-size=4GB -p 8000:8000 \
--userns=keep-id:uid=1001 \
--env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
--env "HF_HUB_OFFLINE=0" \
--env=VLLM_NO_USAGE_STATS=1 \
-v ./rhaiis-cache:/opt/app-root/src/.cache \
registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2 \
--model RedHatAI/Llama-3.2-1B-Instruct-FP8 \
--tensor-parallel-size 2 \
--gpu-memory-utilization 0.8 \
--max-model-len 16384 \
--max-num-batched-tokens 2048 \
```

* [Getting started with Red Hat AI Inference Server](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.2/html-single/getting_started/index)

# vLLM server usage

```terminal
$ vllm [-h] [-v] {chat,complete,serve,bench,collect-env,run-batch}
```

* **chat**\
Generate chat completions via the running API server.
* **complete**\
Generate text completions based on the given prompt via the running API server.
* **serve**\
Start the vLLM OpenAI Compatible API server.
* **bench**\
vLLM bench subcommand.
* **collect-env**\
Start collecting environment information.
* **run-batch**\
Run batch prompts and write results to file.

# vllm chat arguments

Generate chat completions with the running API server.

```terminal
$ vllm chat [options]
```

* **--api-key API_KEY**\
OpenAI API key. If provided, this API key overrides the API key set in the environment variables.

    _Default_: None
* **--model-name MODEL_NAME**\
The model name used in prompt completion, defaults to the first model in list models API call.

    _Default_: None
* **--system-prompt SYSTEM_PROMPT**\
The system prompt to be added to the chat template, used for models that support system prompts.

    _Default_: None
* **--url URL**\
URL of the running OpenAI-compatible RESTful API server

    _Default_: `pass:[[localhost:8000/v1](http://localhost:8000/v1)]`
* **-q MESSAGE, --quick MESSAGE**\
Send a single prompt as `MESSAGE` and print the response, then exit.

    _Default_: None

# vllm complete arguments

Generate text completions based on the given prompt with the running API server.

```terminal
$ vllm complete [options]
```

* **--api-key API_KEY**\
API key for OpenAI services. If provided, this API key overrides the API key set in the environment variables.

    _Default_: None
* **--model-name MODEL_NAME**\
The model name used in prompt completion, defaults to the first model in list models API call.

    _Default_: None
* **--url URL**\
URL of the running OpenAI-compatible RESTful API server

    _Default_: `pass:[[localhost:8000/v1](http://localhost:8000/v1)]`
* **-q PROMPT, --quick PROMPT**\
Send a single prompt and print the completion output, then exit.

    _Default_: None

# vllm serve arguments

Start the vLLM OpenAI compatible API server.

```
$ vllm serve [model_tag] [options]
```

<dl><dt><strong>💡 TIP</strong></dt><dd>

Use `vllm [serve|run-batch] --help=<keyword>` to explore arguments from help:

* To view a argument group: `--help=ModelConfig`
* To view a single argument: `--help=max-num-seqs`
* To search by keyword: `--help=max`
* To list all groups: `--help=listgroup`
</dd></dl>

## Positional arguments

* **model_tag**\
The model tag to serve. Optional if specified in the config.

    _Default_: None

## Options

* **--allow-credentials**\
Allow credentials.

    _Default_: False
* **--allowed-headers ALLOWED_HEADERS**\
Allowed headers.

    _Default_: ['*']
* **--allowed-methods ALLOWED_METHODS**\
Allowed methods.

    _Default_: ['*']
* **--allowed-origins ALLOWED_ORIGINS**\
Allowed origins.

    _Default_: ['*'])
* **--api-key API_KE**\
If provided, the server will require this key to be presented in the
header.

    _Default_: None
* **--api-server-count API_SERVER_COUNT, -asc API_SERVER_COUNT**\
How many API server processes to run.

    _Default_: 1
* **--chat-template CHAT_TEMPLATE**\
The file path to the chat template, or the template in single-line form
for the specified model.

    _Default_: None
* **--chat-template-content-format {auto,string,openai}**

    The format to render message content within a chat template.
    * "string" will render the content as a string. Example: ``"Hello World"``
    * "openai" will render the content as a list of dictionaries, similar to
    OpenAI schema. Example: ``[{"type": "text", "text": "Hello world!"}]``

    _Default_: auto
* **--config CONFIG**\
Read CLI options from a config file.Must be a YAML with the following options: [docs.vllm.ai/en/latest/serving/openai_compatible_server.html#cli-reference](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#cli-reference)

    _Default_:
* **--data-parallel-start-rank DATA_PARALLEL_START_RANK, -dpr DATA_PARALLEL_START_RANK**\
Starting data parallel rank for secondary nodes.

    _Default_: 0
* **--disable-fastapi-docs**\
Disable FastAPI’s OpenAPI schema, Swagger UI, and ReDoc endpoint.

    _Default_: False
* **--disable-frontend-multiprocessing**\
If specified, will run the OpenAI frontend server in the same process as
the model serving engine.

    _Default_: False
* **--disable-log-requests**\
Disable logging requests.

    _Default_: False
* **--disable-log-stats**\
Disable logging statistics.

    _Default_: False
* **--disable-uvicorn-access-log**\
Disable uvicorn access log.

    _Default_: False
* **--enable-auto-tool-choice**\
Enable auto tool choice for supported models. Use ``--tool-call-parser`` to specify which parser to use.

    _Default_: False
* **--enable-prompt-tokens-details**\
If set to True, enable prompt_tokens_details in usage.

    _Default_: False
* **--enable-request-id-headers**\
If specified, API server will add X-Request-Id header to responses.
Caution: this hurts performance at high QPS.

    _Default_: False
* **--enable-server-load-tracking**\
If set to True, enable tracking `server_load_metrics` in the app state.

    _Default_: False
* **--enable-ssl-refresh**\
Refresh SSL Context when SSL certificate files change

    _Default_: False
* **--headless**\
Run in headless mode. See multi-node data parallel documentation for more
details.

    _Default_: False
* **--host HOST**\
Host name.

    _Default_: None
* **--log-config-file LOG_CONFIG_FILE**\
Path to logging config JSON file for both vllm and uvicorn

    _Default_: None
* **--lora-modules LORA_MODULES [LORA_MODULES ...]**\
LoRA module configurations in either `name=path` formator JSON format.
Example (old format): `name=path`.
Example (new format): `{"name": "name", "path": "lora_path", "base_model_name": "id"}`

    _Default_: None
* **--max-log-len MAX_LOG_LEN**\
Max number of prompt characters or prompt ID numbers being printed in log.
The default of None means unlimited.

    _Default_: None
* **--middleware MIDDLEWARE**\
Additional ASGI middleware to apply to the app. We accept multiple `--middleware` arguments.
The value should be an import path. If a function is provided, vLLM will add it to the server using ``@app.middleware('http')``.
If a class is provided, vLLM will add it to the server using ``app.add_middleware()``.

    _Default_: []

--port PORT
Port number.
+
_Default_: 8000

* **--prompt-adapters PROMPT_ADAPTERS [PROMPT_ADAPTERS ...]**\
Prompt adapter configurations in the format name=path.
Multiple adapters can be specified.

    _Default_: None
* **--response-role RESPONSE_ROLE**\
The role name to return if ``request.add_generation_prompt=true``.

    _Default_: assistant
* **--return-tokens-as-token-ids**\
When ``--max-logprobs`` is specified, represents single tokens  as strings of the form 'token_id:{token_id}' so that tokens that are not JSON-encodable can be identified.

    _Default_: False
* **--root-path ROOT_PATH**\
FastAPI root_path when app is behind a path based routing proxy.

    _Default_: None
* **--ssl-ca-certs SSL_CA_CERTS**\
The CA certificates file.

    _Default_: None
* **--ssl-cert-reqs SSL_CERT_REQS**\
Whether client certificate is required (see stdlib ssl module).

    _Default_: 0
* **--ssl-certfile SSL_CERTFILE**\
The file path to the SSL cert file.

    _Default_: None
* **--ssl-keyfile SSL_KEYFILE**\
The file path to the SSL key file.

    _Default_: None
* **--tool-call-parser {deepseek_v3,granite-20b-fc,granite,hermes,internlm,jamba,llama4_pythonic,llama4_json,llama3_json,mistral,phi4_mini_json,pythonic}**\
Or name registered in `--tool-parser-plugin`.
Select the tool call parser depending on the model that you’re using. This is used to parse the model-generated tool call into OpenAI API format.
Required for ``--enable-auto-tool-choice``.

    _Default_: None
* **--tool-parser-plugin TOOL_PARSER_PLUGIN**\
Special the tool parser plugin write to parse the model-generated tool into OpenAI API format, the name register in this plugin can be used in `--tool-call-parser`.

    _Default_:
* **--use-v2-block-manager**\
***DEPRECATED***: block manager v1 has been removed and
SelfAttnBlockSpaceManager (i.e. block manager v2) is now the default.
Setting this flag to True or False has no effect on vLLM behavior.

    _Default_: True
* **--uvicorn-log-level {debug,info,warning,error,critical,trace}**\
Log level for uvicorn.

## Model configuration

* **--allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH**\
Allowing API requests to read local images or videos from directories specified by the server file system. This is a security risk. Should only be enabled in trusted environments.

    _Default_:
* **--code-revision CODE_REVISION**\
The specific revision to use for the model code on the Hugging Face Hub.
It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

    _Default_: None
* **--config-format {auto,hf,mistral}**

    The format of the model config to load:
    * `auto` will try to load the config in hf format if available else it
    will try to load in mistral format.
    * `hf` will load the config in hf format.
    * `mistral` will load the config in mistral format.

    _Default_: auto
* **--disable-async-output-proc**\
Disable async output processing. This may result in lower performance.

    _Default_: False
* **--disable-cascade-attn, --no-disable-cascade-attn**\
Disable cascade attention for V1. While cascade attention does not change the mathematical correctness, disabling it could be useful for preventing potential numerical issues. Note that even if this is set to False, cascade attention will be only used when the heuristic tells that it’s beneficial.

    _Default_: False
* **--disable-sliding-window, --no-disable-sliding-window**\
Whether to disable sliding window. If True, we will disable the sliding window functionality of the model, capping to sliding window size. If the model does not support sliding window, this argument is ignored.

    _Default_: False
* **--dtype {auto,bfloat16,float,float16,float32,half}**

    Data type for model weights and activations:
    * `auto` will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.
    * `half` for FP16. Recommended for AWQ quantization.
    * `float16` is the same as "half".
    * `bfloat16` for a balance between precision and range.
    * `float` is shorthand for FP32 precision.
    * `float32` for FP32 precision.

    _Default_: auto
* **--enable-prompt-embeds, --no-enable-prompt-embeds**\
If `True`, enables passing text embeddings as inputs via the
`prompt_embeds` key. Note that enabling this will double the time required for graph compilation.

    _Default_: False
* **--enable-sleep-mode, --no-enable-sleep-mode**\
Enable sleep mode for the engine (only CUDA platform is supported).

    _Default_: False
* **--enforce-eager, --no-enforce-eager**\
Whether to always use eager-mode PyTorch. If True, we will disable CUDA
graph and always execute the model in eager mode. If False, we will use
CUDA graph and eager execution in hybrid for maximal performance and
flexibility.

    _Default_: False
* **--generation-config GENERATION_CONFIG**\
The folder path to the generation config. Defaults to `auto`, the generation config will be loaded from model path.
If set to `vllm`, no generation config is loaded, vLLM defaults will be used.
If set to a folder path, the generation config will be loaded from the specified folder path.
If `max_new_tokens` is specified in generation config, then it sets a server-wide limit on the number of output tokens for all
requests.

    _Default_: auto
* **--hf-config-path HF_CONFIG_PATH**\
Name or path of the Hugging Face config to use.
If unspecified, model name or path will be used.

    _Default_: None
* **--hf-overrides HF_OVERRIDES**\
If a dictionary, contains arguments to be forwarded to the HuggingFace config.
If a callable, it is called to update the HuggingFace config.

    _Default_: {}
* **--hf-token [HF_TOKEN]**\
The token to use as HTTP bearer authorization for remote files .
If `True`, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).

    _Default_: None
* **--logits-processor-pattern LOGITS_PROCESSOR_PATTERN**\
Optional regex pattern specifying valid logits processor qualified names that can be passed with the `logits_processors` extra completion argument.
Defaults to `None`, which allows no processors.

    _Default_: None
* **--max-logprobs MAX_LOGPROBS**\
Maximum number of log probabilities to return when `logprobs` is specified
in `SamplingParams`.
The default value comes the default for the OpenAI Chat Completions API.

    _Default_: 20

* **--max-model-len MAX_MODEL_LEN**

    Model context length (prompt and output). If unspecified, will be automatically derived from the model config.
    When passing via `--max-model-len`, supports k/m/g/K/M/G in human-readable
    format.
    Examples:
    * 1k -> 1000
    * 1K -> 1024
    * 25.6k -> 25,600

    _Default_: None
* **--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE**\
Maximum sequence len covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode. Additionally for encoder-decoder models, if the sequence length of the encoder input is larger than this, we fall back to the eager mode.

    _Default_: 8192
* **--model MODEL**\
Name or path of the Hugging Face model to use. It is also used as the content for `model_name` tag in metrics output when `served_model_name` is not specified.

    _Default_: facebook/opt-125m
* **--model-impl {auto,vllm,transformers}**

    Which implementation of the model to use:
    * `auto` will try to use the vLLM implementation, if it exists, and fall
    back to the Transformers implementation if no vLLM implementation is
    available.
    * `vllm` will use the vLLM model implementation.
    * `transformers` will use the Transformers model implementation.

    _Default_: auto
* **--override-generation-config OVERRIDE_GENERATION_CONFIG**

    Overrides or sets generation config. e.g. `{"temperature": 0.5}`.
    If used with `--generation-config auto`, the override parameters will be merged with the default config from the model.
    If used with `--generation-config vllm`, only the override parameters are used.
    Should either be a valid JSON string or JSON keys passed individually.
    For example, the following sets of arguments are equivalent:
    * `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
    * `--json-arg.key1 value1 --json-arg.key2.key3 value2`

    _Default_: {}
* **--override-neuron-config OVERRIDE_NEURON_CONFIG**

    Initialize non-default neuron config or override default neuron config that are specific to Neuron devices, this argument will be used to configure the neuron config that can not be gathered from the vllm arguments. e.g. `{"cast_logits_dtype": "bfloat16"}`.
    Should either be a valid JSON string or JSON keys passed individually. For example, the following sets of arguments are equivalent:
    * `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
    * `--json-arg.key1 value1 --json-arg.key2.key3 value2`

    _Default_: {}
* **--override-pooler-config OVERRIDE_POOLER_CONFIG**\
Initialize non-default pooling config or override default pooling config for the pooling model.
For example, `{"pooling_type": "mean", "normalize": false}`.

    _Default_: None
* **--quantization {aqlm,auto-round,awq, ...}, -q**\
Method used to quantize the weights. If `None`, we first check the `quantization_config` attribute in the model config file. If that is `None`, we assume the model weights are not quantized and use `dtype` to determine the data type of the weights.

    _Default_: None
* **--revision REVISION**\
The specific model version to use. It can be branch name, a tag name, or a commit id. If unspecified, will use the default version.

    _Default_: None
* **--rope-scaling ROPE_SCALING**

    RoPE scaling configuration. For example, `{"rope_type":"dynamic","factor":2.0}`.
    Should either be a valid JSON string or JSON keys passed individually. For example, the following sets of arguments are equivalent:
    * `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
    * `--json-arg.key1 value1 --json-arg.key2.key3 value2`

    _Default_: {}
* **--rope-theta ROPE_THETA**\
RoPE theta. Use with `rope_scaling`. In some cases, changing the RoPE theta improves the performance of the scaled model.

    _Default_: None
* **--seed SEED**\
Random seed for reproducibility. Initialized to None in V0, but initialized to 0 in V1.

    _Default_: None
* **--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]**\
The model name(s) used in the API. If multiple names are provided, the server will respond to any of the provided names.
The model name in the model field of a response will be the first name in this list. If not specified, the model name will be the same as the `--model` argument.
Noted that this name(s) will also be used in `model_name` tag content of prometheus metrics, if multiple names provided, metrics tag will take the first one.

    _Default_: None
* **--skip-tokenizer-init, --no-skip-tokenizer-init**\
Skip initialization of tokenizer and detokenizer. Expects valid `prompt_token_ids` and `None` for prompt from the input. The generated output will contain token ids.

    _Default_: False
* **--task {auto,classify,draft,embed,embedding,generate,reward,score,transcription}**\
The task to use the model for. Each vLLM instance only supports one task, even if the same model can be used for multiple tasks. When the model only supports one task, `auto` can be used to select it; otherwise, you must specify explicitly which task to use.

    _Default_: auto
* **--tokenizer TOKENIZER**\
Name or path of the Hugging Face tokenizer to use. If unspecified, model name or path will be used.

    _Default_: None
* **--tokenizer-mode {auto,custom,mistral,slow}**

    Tokenizer mode:
    * `auto` will use the fast tokenizer if available.
    * `slow` will always use the slow tokenizer.
    * `mistral` will always use the tokenizer from `mistral_common`.
    * `custom` will use --tokenizer to select the preregistered tokenizer.

    _Default_: auto
* **--tokenizer-revision TOKENIZER_REVISION**\
The specific revision to use for the tokenizer on the Hugging Face Hub.
It can be a branch name, a tag name, or a commit id.
If unspecified, will use the default version.

    _Default_: None
* **--trust-remote-code, --no-trust-remote-code**\
Trust remote code (e.g., from HuggingFace) when downloading the model and
tokenizer.

    _Default_: False

## Model load configuration

Configuration for loading the model weights.

* **--download-dir DOWNLOAD_DIR**\
Directory to download and load the weights, default to the default cache
directory of Hugging Face.

    _Default_: None
* **--ignore-patterns IGNORE_PATTERNS [IGNORE_PATTERNS ...]**\
The list of patterns to ignore when loading the model. Defaults to
`"original/***/**"` to avoid repeated loading of llama’s checkpoints.

    _Default_: None
* **--load-format {auto,pt,safetensors, ...}**

    The format of the model weights to load:

    * `auto`: will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available.
    * `pt`: will load the weights in the pytorch bin format.
    * `safetensors`: will load the weights in the safetensors format.
    * `npcache`: will load the weights in pytorch format and store a numpy cache to speed up the loading.
    * `dummy`: will initialize the weights with random values, which is mainly for profiling.
    * `tensorizer`: will use CoreWeave’s tensorizer library for fast weight loading. See the Tensorize vLLM Model script in the Examples section for more information.
    * `runai_streamer`: will load the Safetensors weights using Run:ai Model Streamer.
    * `bitsandbytes`: will load the weights using bitsandbytes quantization.
    * `sharded_state`: will load weights from pre-sharded checkpoint files, supporting efficient loading of tensor-parallel models.
    * `gguf`: will load weights from GGUF format files (details specified in [github.com/ggml-org/ggml/blob/master/docs/gguf.md).](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).)
    * `mistral`: will load weights from consolidated safetensors files used by Mistral models.

    _Default_: auto
* **--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG**\
Extra config for model loader. This will be passed to the model loader corresponding to the chosen `load_format`.

    _Default_: {}
* **--pt-load-map-location PT_LOAD_MAP_LOCATION**\
`PT_LOAD_MAP_LOCATION`: the map location for loading pytorch checkpoint, to support loading checkpoints can only be loaded on certain devices like "CUDA", this is equivalent to `{"": "CUDA"}`.
Another supported format is mapping from different devices like from GPU 1 to GPU 0: `{"CUDA:1": "CUDA:0"}`. Note that when passed from command line, the strings in dictionary needs to be double quoted for JSON parsing. For more details, see original doc for `map_location` in [pytorch.org/docs/stable/generated/torch.load.html](https://pytorch.org/docs/stable/generated/torch.load.html)

    _Default_: cpu
* **--qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH**\
The `--qlora-adapter-name-or-path` has no effect, do not set it, and it will be removed in v0.10.0.

    _Default_: None
* **--use-tqdm-on-load, --no-use-tqdm-on-load**\
Whether to enable tqdm for showing progress bar when loading model weights.

    _Default_: True

## Decoding configuration

Data class that contains the decoding strategy for the engine.

* **--enable-reasoning, --no-enable-reasoning**\
[DEPRECATED] The `--enable-reasoning` flag is deprecated as of v0.9.0.
Use `--reasoning-parser` to specify the reasoning parser backend instead.
This flag (`--enable-reasoning`) will be removed in v0.10.0. When `--reasoning-parser` is specified, reasoning mode is automatically enabled.

    _Default_: None
* **--guided-decoding-backend {auto,guidance,lm-format-enforcer,outlines,xgrammar}**\
Which engine will be used for guided decoding (JSON schema / regex etc) by default. With `auto`, we will make opinionated choices based on request contents and what the backend libraries currently support, so the behavior is subject to change in each release.

    _Default_: auto
* **--guided-decoding-disable-additional-properties, --no-guided-decoding-disable-additional-properties**\
If `True`, the `guidance` backend will not use `additionalProperties` in
the JSON schema. This is only supported for the `guidance` backend and is
used to better align its behaviour with `outlines` and `xgrammar`.

    _Default_: False
* **--guided-decoding-disable-any-whitespace, --no-guided-decoding-disable-any-whitespace**\
If `True`, the model will not generate any whitespace during guided
decoding. This is only supported for xgrammar and guidance backends.

    _Default_: False
* **--guided-decoding-disable-fallback, --no-guided-decoding-disable-fallback**\
If `True`, vLLM will not fallback to a different backend on error.

    _Default_: False
* **--reasoning-parser {deepseek_r1,granite,qwen3}**\
Select the reasoning parser depending on the model that you’re using. This
is used to parse the reasoning content into OpenAI API format.

    _Default_:

## Parallel configuration

Configuration for the distributed execution.

* **--data-parallel-address DATA_PARALLEL_ADDRESS, -dpa DATA_PARALLEL_ADDRESS**\
Address of data parallel cluster head-node.

    _Default_: None
* **--data-parallel-backend DATA_PARALLEL_BACKEND, -dpb DATA_PARALLEL_BACKEND**\
Backend for data parallel, either `mp` or `ray`.

    _Default_: mp
* **--data-parallel-rpc-port DATA_PARALLEL_RPC_PORT, -dpp DATA_PARALLEL_RPC_PORT**\
Port for data parallel RPC communication.

    _Default_: None
* **--data-parallel-size DATA_PARALLEL_SIZE, -dp DATA_PARALLEL_SIZE**\
Number of data parallel groups. MoE layers will be sharded according to
the product of the tensor parallel size and data parallel size.

    _Default_: 1
* **--data-parallel-size-local DATA_PARALLEL_SIZE_LOCAL, -dpl DATA_PARALLEL_SIZE_LOCAL**\
Number of data parallel replicas to run on this node.

    _Default_: None
* **--disable-custom-all-reduce, --no-disable-custom-all-reduce**\
Disable the custom all-reduce kernel and fall back to NCCL.

    _Default_: False
* **--distributed-executor-backend {external_launcher,mp,ray,uni,None}**\
Backend to use for distributed model workers, either `ray` or `mp` (multiprocessing). If the product of `pipeline_parallel_size` and `tensor_parallel_size` is less than or equal to the number of GPUs available, `mp` will be used to keep processing on a single host.
Otherwise, this will default to `ray` if Ray is installed and fail otherwise.
Note that TPU and HPU only support Ray for distributed inference.

    _Default_: None
* **--enable-expert-parallel, --no-enable-expert-parallel**\
Use expert parallelism instead of tensor parallelism for MoE layers.

    _Default_: False
* **--enable-multimodal-encoder-data-parallel, --no-enable-multimodal-encoder-data-parallel**\
Use data parallelism instead of tensor parallelism for vision encoder. Only support LLama4 for now

    _Default_: False
* **--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS**\
Maximum number of parallel loading workers when loading model sequentially in multiple batches.
To avoid RAM OOM when using tensor parallel and large models.

    _Default_: None
* **--pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE**\
Number of pipeline parallel groups.

    _Default_: 1
* **--ray-workers-use-nsight, --no-ray-workers-use-nsight**\
Whether to profile Ray workers with nsight, see [docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.](https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.)

    _Default_: False

* **--tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE**\
Number of tensor parallel groups.

    <a name="tensor-parallel-size"></a>_Default_: 1
* **--worker-cls WORKER_CLS**\
The full name of the worker class to use. If `auto`, the worker class will be determined based on the platform.

    _Default_: auto
* **--worker-extension-cls WORKER_EXTENSION_CLS**\
The full name of the worker extension class to use. The worker extension class is dynamically inherited by the worker class. This is used to inject new attributes and methods to the worker class for use in `collective_rpc` calls.

    _Default_:

## Cache configuration

Configuration for the KV cache.

* **--block-size {1,8,16,32,64,128}**\
Size of a contiguous cache block in number of tokens. This is ignored on neuron devices and set to `--max-model-len`. On CUDA devices, only block sizes up to 32 are supported. On HPU devices, block size defaults to 128. This config has no static default. If left unspecified by the user, it will be set in `Platform.check_and_update_configs()` based on the current platform.

    _Default_: None
* **--calculate-kv-scales, --no-calculate-kv-scales**\
This enables dynamic calculation of `k_scale` and `v_scale` when `kv_cache_dtype` is `fp8`. If `False`, the scales will be loaded from the model checkpoint if available. Otherwise, the scales will default to 1.0.

    _Default_: False
* **--cpu-offload-gb CPU_OFFLOAD_GB**\
The space in GiB to offload to CPU, per GPU. Default is 0, which means no offloading. Intuitively, this argument can be seen as a virtual way to increase the GPU memory size. For example, if you have one 24 GB GPU and set this to 10, virtually you can think of it as a 34 GB GPU.
Then you can load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
Note that this requires fast CPU-GPU interconnect, as part of the model is loaded from CPU memory to GPU memory on the fly in each model
forward pass.

    _Default_: 0
* **--enable-prefix-caching, --no-enable-prefix-caching**\
Whether to enable prefix caching. Disabled by default for V0. Enabled by default for V1.

    _Default_: None

* **--gpu-memory-utilization GPU_MEMORY_UTILIZATION**\
The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.
For example, a value of 0.5 would imply 50% GPU memory utilization. If unspecified, will use the default value of 0.9.
This is a per-instance limit, and only applies to the current vLLM instance.
It does not matter if you have another vLLM instance running on the same GPU. For example, if you have two vLLM instances running on the same GPU, you can set the GPU memory utilization to 0.5 for each instance.

    <a name="gpu-memory-utilization"></a>_Default_: 0.9
* **--kv-cache-dtype {auto,fp8,fp8_e4m3,fp8_e5m2}**\
Data type for kv cache storage.
If `auto`, will use model data type.
CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports fp8 (=fp8_e4m3).

    _Default_: auto
* **--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE**\
Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
if specified.
Does nothing if `None`. Used for testing preemption.

    _Default_: None
* **--prefix-caching-hash-algo {builtin,sha256}**

    Set the hash algorithm for prefix caching:
    * "builtin" is Python’s built-in hash.
    * "sha256" is collision resistant but with certain overheads.

    _Default_: builtin
* **--swap-space SWAP_SPACE**\
Size of the CPU swap space per GPU (in GiB).

    _Default_: 4

## Multi-modal model configuration
Controls the behavior of multi-modal models.

* **--disable-mm-preprocessor-cache, --no-disable-mm-preprocessor-cache**\
If `True`, disable caching of the processed multi-modal inputs.

    _Default_: False
* **--limit-mm-per-prompt LIMIT_MM_PER_PROMPT**

    The maximum number of input items allowed per prompt for each modality.
    Defaults to 1 (V0) or 999 (V1) for each modality.
    For example, to allow up to 16 images and 2 videos per prompt: `{"images": 16, "videos": 2}`
    Should either be a valid JSON string or JSON keys passed individually.
    For example, the following sets of arguments are equivalent:
    * `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
    * `--json-arg.key1 value1 --json-arg.key2.key3 value2`

    _Default_: {}
* **--mm-processor-kwargs MM_PROCESSOR_KWARGS**

    Overrides for the multi-modal processor obtained from `transformers.AutoProcessor.from_pretrained`.
    The available overrides depend on the model that is being run.
    For example, for Phi-3-Vision: `{"num_crops": 4}`.
    Should either be a valid JSON string or JSON keys passed individually. For example, the following sets of arguments are equivalent:
    * `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
    * `--json-arg.key1 value1 --json-arg.key2.key3 value2`

    _Default_: None

## LoRA configuration

* **--enable-lora, --no-enable-lora**\
If True, enable handling of LoRA adapters.

    _Default_: None
* **--enable-lora-bias, --no-enable-lora-bias**\
Enable bias for LoRA adapters.

    _Default_: False
* **--fully-sharded-loras, --no-fully-sharded-loras**\
By default, only half of the LoRA computation is sharded with tensor parallelism. Enabling this will use the fully sharded layers.
At high sequence length, max rank or tensor parallel size, this is likely faster.

    _Default_: False
* **--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS [LONG_LORA_SCALING_FACTORS ...]**\
Specify multiple scaling factors (which can be different from base model scaling factor) to allow for multiple LoRA adapters trained with those scaling factors to be used at the same time.
If not specified, only adapters trained with the base model scaling factor are allowed.

    _Default_: None
* **--lora-dtype {auto,bfloat16,float16}**\
Data type for LoRA. If auto, will default to base model dtype.

    _Default_: auto
* **--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE**\
Maximum size of extra vocabulary that can be present in a LoRA adapter (added to the base model vocabulary).

    _Default_: 256
* **--max-cpu-loras MAX_CPU_LORAS**\
Maximum number of LoRAs to store in CPU memory. Must be greater than `max_loras`.

    _Default_: None
* **--max-lora-rank MAX_LORA_RANK**\
Max LoRA rank.

    _Default_: 16
* **--max-loras MAX_LORAS**\
Max number of LoRAs in a single batch.

    _Default_: 1

## Prompt adapter configuration

* **--enable-prompt-adapter, --no-enable-prompt-adapter**\
If True, enable handling of PromptAdapters.

    _Default_: None
* **--max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN**\
Max number of PromptAdapters tokens.

    _Default_: 0
* **--max-prompt-adapters MAX_PROMPT_ADAPTERS**\
Max number of PromptAdapters in a batch.

    _Default_: 1

## Device configuration

* **--device {auto,cpu,CUDA,hpu,neuron,tpu,xpu}**\
Device type for vLLM execution. This parameter is deprecated and will be removed in a future release. It will now be set automatically based on the current platform.

    _Default_: auto

## Speculative decoding configuration

* **--speculative-config SPECULATIVE_CONFIG**\
The configurations for speculative decoding. Should be a JSON string.

    _Default_: None

## Observability configuration

* **--collect-detailed-traces {all,model,worker,None} [{all,model,worker,None} ...]**\
It makes sense to set this only if `--otlp-traces-endpoint` is set. If set, it will collect detailed traces for the specified modules. This involves use of possibly costly and or blocking operations and hence might have a performance impact.
Note that collecting detailed timing information for each request can be expensive.

    _Default_: None
* **--otlp-traces-endpoint OTLP_TRACES_ENDPOINT**\
Target URL to which OpenTelemetry traces will be sent.

    _Default_: None
* **--show-hidden-metrics-for-version SHOW_HIDDEN_METRICS_FOR_VERSION**\
Enable deprecated Prometheus metrics that have been hidden since the specified version. For example, if a previously deprecated metric has been hidden since the v0.7.0 release, you use `--show-hidden-metrics-for-version=0.7` as a temporary escape hatch while you migrate to new metrics.
The metric is likely to be removed completely in an upcoming release.

    _Default_: None

## Scheduler configuration

* **--CUDA-graph-sizes CUDA_GRAPH_SIZES [CUDA_GRAPH_SIZES ...]**\
Cuda graph capture sizes, default is 512. 1. if one value is provided, then the capture list would follow the pattern: `[1, 2, 4] + [i for i in range(8, CUDA_graph_sizes + 1, 8)] 2`. more than one value (e.g. 1 2 128) is provided, then the capture list will follow the provided list.

    _Default_: 512
* **--disable-chunked-mm-input, --no-disable-chunked-mm-input**\
If set to true and chunked prefill is enabled, we do not want to partially schedule a multimodal item.
Only used in V1 This ensures that if a request has a mixed prompt (like text tokens TTTT followed by image tokens IIIIIIIIII) where only some image tokens can be scheduled (like TTTTIIIII, leaving IIIII), it will be scheduled as TTTT in one step and IIIIIIIIII in the next.

    _Default_: False
* **--disable-hybrid-kv-cache-manager, --no-disable-hybrid-kv-cache-manager**\
If set to True, KV cache manager will allocate the same size of KV cache for all attention layers even if there are multiple type of attention layers like full attention and sliding window attention.

    _Default_: False
* **--enable-chunked-prefill, --no-enable-chunked-prefill**\
If True, prefill requests can be chunked based on the remaining `max_num_batched_tokens`.

    _Default_: None
* **--long-prefill-token-threshold LONG_PREFILL_TOKEN_THRESHOLD**\
For chunked prefill, a request is considered long if the prompt is longer than this number of tokens.

    _Default_: 0
* **--max-long-partial-prefills MAX_LONG_PARTIAL_PREFILLS**\
For chunked prefill, the maximum number of prompts longer than `long_prefill_token_threshold` that will be prefilled concurrently. Setting this less than `max_num_partial_prefills` will allow shorter prompts to jump the queue in front of longer prompts in some cases, improving latency.

    _Default_: 1

* **--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS**\
Maximum number of tokens to be processed in a single iteration.
This config has no static default. If left unspecified by the user, it will be set in `EngineArgs.create_engine_config` based on the usage context.

    <a name="max-num-batched-tokens"></a>_Default_: None
* **--max-num-partial-prefills MAX_NUM_PARTIAL_PREFILLS**\
For chunked prefill, the maximum number of sequences that can be partially prefilled concurrently.

    _Default_: 1
* **--max-num-seqs MAX_NUM_SEQS**\
Maximum number of sequences to be processed in a single iteration.
This config has no static default. If left unspecified by the user, it will be set in `EngineArgs.create_engine_config` based on the usage context.

    _Default_: None
* **--multi-step-stream-outputs, --no-multi-step-stream-outputs**\
If False, then multi-step will stream outputs at the end of all steps

    _Default_: True
* **--num-lookahead-slots NUM_LOOKAHEAD_SLOTS**

    The number of slots to allocate per sequence per step, beyond the known token ids. This is used in speculative decoding to store KV activations of tokens which may or may not be accepted.

    <dl><dt><strong>📌 NOTE</strong></dt><dd>

    This will be replaced by speculative config in the future; it is present to enable correctness tests until then.
    </dd></dl>

   \
    _Default_: 0
* **--num-scheduler-steps NUM_SCHEDULER_STEPS**\
Maximum number of forward steps per scheduler call.

    _Default_: 1
* **--preemption-mode {recompute,swap,None}**\
Whether to perform preemption by swapping or recomputation. If not specified, we determine the mode as follows: We use recomputation by default since it incurs lower overhead than swapping. However, when the sequence group has multiple sequences (e.g., beam search), recomputation is not currently supported. In such a case, we use swapping instead.

    _Default_: None
* **--scheduler-cls SCHEDULER_CLS**\
The scheduler class to use. "vllm.core.scheduler.Scheduler" is the default scheduler. Can be a class directly or the path to a class of form "mod.custom_class".

    _Default_: vllm.core.scheduler.Scheduler
* **--scheduler-delay-factor SCHEDULER_DELAY_FACTOR**\
Apply a delay (of delay factor multiplied by previous prompt latency) before scheduling next prompt.

    _Default_: 0.0
* **--scheduling-policy {fcfs,priority}**

    The scheduling policy to use:
    * "fcfs" means first come first served, i.e. requests are handled in order of arrival.
    * "priority" means requests are handled based on given priority (lower value means earlier handling) and time of arrival deciding any ties).

    _Default_: fcfs

## vllm configuration

* **--additional-config ADDITIONAL_CONFIG**\
Additional config for specified platform. Different platforms may support different configs. Make sure the configs are valid for the platform you are using. Contents must be hashable.

    _Default_: {}
* **--compilation-config COMPILATION_CONFIG, -O COMPILATION_CONFIG**

    `torch.compile` configuration for the model.
    When it is a number (0, 1, 2, 3), it will be interpreted as the
    optimization level.

    <dl><dt><strong>📌 NOTE</strong></dt><dd>

    level 0 is the default level without any optimization. level 1 and 2 are for internal testing only. level 3 is the recommended level for production.
    </dd></dl>

    Following the convention of traditional compilers, using `-O` without space is also supported. `-O3` is equivalent to `-O 3`.
    You can specify the full compilation config like so: `{"level": 3, "CUDAgraph_capture_sizes": [1, 2, 4, 8]}`.
    Should either be a valid JSON string or JSON keys passed individually.
    For example, the following sets of arguments are equivalent:
    * `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
    * `--json-arg.key1 value1 --json-arg.key2.key3 value2`

    _Default_:
    ```json
    {
      "level": 0,
      "debug_dump_path": "",
      "cache_dir": "",
      "backend": "",
      "custom_ops": [],
      "splitting_ops": [],
      "use_inductor": true,
      "compile_sizes": null,
      "inductor_compile_config": {
        "enable_auto_functionalized_v2": false
      },
      "inductor_passes": {},
      "use_CUDAgraph": true,
      "CUDAgraph_num_of_warmups": 0,
      "CUDAgraph_capture_sizes": null,
      "CUDAgraph_copy_inputs": false,
      "full_CUDA_graph": false,
      "max_capture_size": null,
      "local_cache_dir": null
    }
    ```
* **--kv-events-config KV_EVENTS_CONFIG**

    The configurations for event publishing.
    Should either be a valid JSON string or JSON keys passed individually. For
    example, the following sets of arguments are equivalent:
    * `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
    * `--json-arg.key1 value1 --json-arg.key2.key3 value2`

    _Default_: None
* **--kv-transfer-config KV_TRANSFER_CONFIG**

    The configurations for distributed KV cache transfer.
    Should either be a valid JSON string or JSON keys passed individually. For example, the following sets of arguments are equivalent:
    * `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
    * `--json-arg.key1 value1 --json-arg.key2.key3 value2`

    _Default_: None

# vllm bench arguments

Benchmark online serving throughput.

```terminal
$ vllm bench [options]
```

* **bench**\
Positional arguments:
    * `latency` - Benchmarks the latency of a single batch of requests.
    * `serve` - Benchmarks the online serving throughput.
    * `throughput` - Benchmarks offline inference throughput.

# vllm collect-env arguments

Collect environment information.

```terminal
$ vllm collect-env
```

# vllm run-batch arguments

Run batch inference jobs for the specified model.

```terminal
$ vllm run-batch
```

* **--disable-log-requests**\
Disable logging requests.

    _Default_: False
* **--disable-log-stats**\
Disable logging statistics.

    _Default_: False
* **--enable-metrics**\
Enables Prometheus metrics.

    _Default_: False
* **--enable-prompt-tokens-details**\
Enables `prompt_tokens_details` in usage when set to True.

    _Default_: False
* **--max-log-len MAX_LOG_LEN**\
Maximum number of prompt characters or prompt ID numbers printed in the log.

    _Default_: Unlimited
* **--output-tmp-dir OUTPUT_TMP_DIR**\
The directory to store the output file before uploading it to the output
URL.

    _Default_: None
* **--port PORT**\
Port number for the Prometheus metrics server.
Only needed if `enable-metrics` is set.

    _Default_: 8000
* **--response-role RESPONSE_ROLE**\
The role name to return if `request.add_generation_prompt=True`.

    _Default_: assistant
* **--url URL**\
Prometheus metrics server URL.
Only required if `enable-metrics` is set).

    _Default_: 0.0.0.0
* **--use-v2-block-manager**\
***DEPRECATED***. Block manager v1 has been removed. `SelfAttnBlockSpaceManager` (block manager v2) is now the default.
Setting `--use-v2-block-manager` flag to True or False has no effect on vLLM behavior.

    _Default_: True
* **-i INPUT_FILE, --input-file INPUT_FILE**\
The path or URL to a single input file.
Supports local file paths and HTTP or HTTPS.
If a URL is specified, the file should be available using HTTP GET.

    _Default_: None
* **-o OUTPUT_FILE, --output-file OUTPUT_FILE**\
The path or URL to a single output file.
Supports local file paths and HTTP or HTTPS.
If a URL is specified, the file should be available using HTTP PUT.

    _Default_: None

# Environment variables

You can use environment variables to configure the system-level installation, build, logging behavior of AI Inference Server.

<dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

`VLLM_PORT` and `VLLM_HOST_IP` set the host ports and IP address for **internal usage** of AI Inference Server. It is not the port and IP address for the API server.
Do not use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server.
</dd></dl>

<dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

All environment variables used by AI Inference Server are prefixed with `VLLM_`.
If you are using Kubernetes, do not name the service `vllm`, otherwise environment variables set by Kubernetes might come into conflict with AI Inference Server environment variables.
This is because Kubernetes sets environment variables for each service with the capitalized service name as the prefix. For more information, see [Kubernetes environment variables](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).
</dd></dl>

**AI Inference Server environment variables**

| Environment variable | Description |
| --- | --- |
| `VLLM_TARGET_DEVICE` | Target device of vLLM, supporting `cuda` (by default), `rocm`, `neuron`, `cpu`, `openvino`. |
| `MAX_JOBS` | Maximum number of compilation jobs to run in parallel. By default, this is the number of CPUs. |
| `NVCC_THREADS` | Number of threads to use for nvcc. By default, this is 1. If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU. |
| `VLLM_USE_PRECOMPILED` | If set, AI Inference Server uses precompiled binaries (\*.so). |
| `VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL` | Whether to force using nightly wheel in Python build for testing. |
| `CMAKE_BUILD_TYPE` | CMake build type. Available options: "Debug", "Release", "RelWithDebInfo". |
| `VERBOSE` | If set, AI Inference Server prints verbose logs during installation. |
| `VLLM_CONFIG_ROOT` | Root directory for AI Inference Server configuration files. |
| `VLLM_CACHE_ROOT` | Root directory for AI Inference Server cache files. |
| `VLLM_HOST_IP` | Used in a distributed environment to determine the IP address of the current node. |
| `VLLM_PORT` | Used in a distributed environment to manually set the communication port. |
| `VLLM_RPC_BASE_PATH` | Path used for IPC when the frontend API server is running in multi-processing mode. |
| `VLLM_USE_MODELSCOPE` | If true, will load models from ModelScope instead of Hugging Face Hub. |
| `VLLM_RINGBUFFER_WARNING_INTERVAL` | Interval in seconds to log a warning message when the ring buffer is full. |
| `CUDA_HOME` | Path to cudatoolkit home directory, under which should be bin, include, and lib directories. |
| `VLLM_NCCL_SO_PATH` | Path to the NCCL library file. Needed for versions of NCCL >= 2.19 due to a bug in PyTorch. |
| `LD_LIBRARY_PATH` | Used when `VLLM_NCCL_SO_PATH` is not set, AI Inference Server tries to find the NCCL library in this path. |
| `VLLM_USE_TRITON_FLASH_ATTN` | Flag to control if you wantAI Inference Server to use Triton Flash Attention. |
| `VLLM_FLASH_ATTN_VERSION` | Force AI Inference Server to use a specific flash-attention version (2 or 3), only valid with the flash-attention backend. |
| `VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE` | Internal flag to enable Dynamo fullgraph capture. |
| `LOCAL_RANK` | Local rank of the process in the distributed setting, used to determine the GPU device ID. |
| `CUDA_VISIBLE_DEVICES` | Used to control the visible devices in a distributed setting. |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | Timeout for each iteration in the engine. |
| `VLLM_API_KEY` | API key for AI Inference Server API server. |
| `S3_ACCESS_KEY_ID` | S3 access key ID for tensorizer to load model from S3. |
| `S3_SECRET_ACCESS_KEY` | S3 secret access key for tensorizer to load model from S3. |
| `S3_ENDPOINT_URL` | S3 endpoint URL for tensorizer to load model from S3. |
| `VLLM_USAGE_STATS_SERVER` | URL for AI Inference Server usage stats server. |
| `VLLM_NO_USAGE_STATS` | If true, disables collection of usage stats. |
| `VLLM_DO_NOT_TRACK` | If true, disables tracking of AI Inference Server usage stats. |
| `VLLM_USAGE_SOURCE` | Source for usage stats collection. |
| `VLLM_CONFIGURE_LOGGING` | If set to 1, AI Inference Server configures logging using the default configuration or the specified config path. |
| `VLLM_LOGGING_CONFIG_PATH` | Path to the logging configuration file. |
| `VLLM_LOGGING_LEVEL` | Default logging level for vLLM. |
| `VLLM_LOGGING_PREFIX` | If set, AI Inference Server prepends this prefix to all log messages. |
| `VLLM_LOGITS_PROCESSOR_THREADS` | Number of threads used for custom logits processors. |
| `VLLM_TRACE_FUNCTION` | If set to 1, AI Inference Server traces function calls for debugging. |
| `VLLM_ATTENTION_BACKEND` | Backend for attention computation, for example , "TORCH_SDPA", "FLASH_ATTN", "XFORMERS"). |
| `VLLM_USE_FLASHINFER_SAMPLER` | If set, AI Inference Server uses the FlashInfer sampler. |
| `VLLM_FLASHINFER_FORCE_TENSOR_CORES` | Forces FlashInfer to use tensor cores; otherwise uses heuristics. |
| `VLLM_PP_LAYER_PARTITION` | Pipeline stage partition strategy. |
| `VLLM_CPU_KVCACHE_SPACE` | CPU key-value cache space (default is 4GB). |
| `VLLM_CPU_OMP_THREADS_BIND` | CPU core IDs bound by OpenMP threads. |
| `VLLM_CPU_MOE_PREPACK` | Whether to use prepack for MoE layer on unsupported CPUs. |
| `VLLM_OPENVINO_DEVICE` | OpenVINO device selection (default is CPU). |
| `VLLM_OPENVINO_KVCACHE_SPACE` | OpenVINO key-value cache space (default is 4GB). |
| `VLLM_OPENVINO_CPU_KV_CACHE_PRECISION` | Precision for OpenVINO KV cache. |
| `VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS` | Enables weights compression during model export by using HF Optimum. |
| `VLLM_USE_RAY_SPMD_WORKER` | Enables Ray SPMD worker for execution on all workers. |
| `VLLM_USE_RAY_COMPILED_DAG` | Uses the Compiled Graph API provided by Ray to optimize control plane overhead. |
| `VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL` | Enables NCCL communication in the Compiled Graph provided by Ray. |
| `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` | Enables GPU communication overlap in the Compiled Graph provided by Ray. |
| `VLLM_WORKER_MULTIPROC_METHOD` | Specifies the method for multiprocess workers, for example, "fork"). |
| `VLLM_ASSETS_CACHE` | Path to the cache for storing downloaded assets. |
| `VLLM_IMAGE_FETCH_TIMEOUT` | Timeout for fetching images when serving multimodal models (default is 5 seconds). |
| `VLLM_VIDEO_FETCH_TIMEOUT` | Timeout for fetching videos when serving multimodal models (default is 30 seconds). |
| `VLLM_AUDIO_FETCH_TIMEOUT` | Timeout for fetching audio when serving multimodal models (default is 10 seconds). |
| `VLLM_MM_INPUT_CACHE_GIB` | Cache size in GiB for multimodal input cache (default is 8GiB). |
| `VLLM_XLA_CACHE_PATH` | Path to the XLA persistent cache directory (only for XLA devices). |
| `VLLM_XLA_CHECK_RECOMPILATION` | If set, asserts on XLA recompilation after each execution step. |
| `VLLM_FUSED_MOE_CHUNK_SIZE` | Chunk size for fused MoE layer (default is 32768). |
| `VLLM_NO_DEPRECATION_WARNING` | If true, skips deprecation warnings. |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | If true, keeps the OpenAI API server alive even after engine errors. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Allows specifying a max sequence length greater than the default length of the model. |
| `VLLM_TEST_FORCE_FP8_MARLIN` | Forces FP8 Marlin for FP8 quantization regardless of hardware support. |
| `VLLM_TEST_FORCE_LOAD_FORMAT` | Forces a specific load format. |
| `VLLM_RPC_TIMEOUT` | Timeout for fetching response from backend server. |
| `VLLM_PLUGINS` | List of plugins to load. |
| `VLLM_TORCH_PROFILER_DIR` | Directory for saving Torch profiler traces. |
| `VLLM_USE_TRITON_AWQ` | If set, uses Triton implementations of AWQ. |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | If set, allows updating Lora adapters at runtime. |
| `VLLM_SKIP_P2P_CHECK` | Skips peer-to-peer capability check. |
| `VLLM_DISABLED_KERNELS` | List of quantization kernels to disable for performance comparisons. |
| `VLLM_USE_V1` | If set, uses V1 code path. |
| `VLLM_ROCM_FP8_PADDING` | Pads FP8 weights to 256 bytes for ROCm. |
| `Q_SCALE_CONSTANT` | Divisor for dynamic query scale factor calculation for FP8 KV Cache. |
| `K_SCALE_CONSTANT` | Divisor for dynamic key scale factor calculation for FP8 KV Cache. |
| `V_SCALE_CONSTANT` | Divisor for dynamic value scale factor calculation for FP8 KV Cache. |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | If set, enables multiprocessing in LLM for the V1 code path. |
| `VLLM_LOG_BATCHSIZE_INTERVAL` | Time interval for logging batch size. |
| `VLLM_SERVER_DEV_MODE` | If set, AI Inference Server runs in development mode, enabling additional endpoints for debugging, for example `/reset_prefix_cache`). |
| VLLM_V1_OUTPUT_PROC_CHUNK_SIZE | Controls the maximum number of requests to handle in a single asyncio task for processing per-token outputs in the V1 AsyncLLM interface. It affects high-concurrency streaming requests. |
| `VLLM_MLA_DISABLE` | If set, AI Inference Server disables the MLA attention optimizations. |
| `VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON` | If set, AI Inference Server uses the Triton implementation of `moe_align_block_size`, for example, `moe_align_block_size_triton` in `fused_moe.py`. |
| `VLLM_RAY_PER_WORKER_GPUS` | Number of GPUs per worker in Ray. Can be a fraction to allow Ray to schedule multiple actors on a single GPU. |
| `VLLM_RAY_BUNDLE_INDICES` | Specifies the indices used for the Ray bundle, for each worker. Format: comma-separated list of integers (e.g., "0,1,2,3"). |
| `VLLM_CUDART_SO_PATH` | Specifies the path for the `find_loaded_library()` method when it may not work properly. Set by using the `VLLM_CUDART_SO_PATH` environment variable. |
| `VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH` | Enables contiguous cache fetching to avoid costly gather operations on Gaudi3. Only applicable to HPU contiguous cache. |
| `VLLM_DP_RANK` | Rank of the process in the data parallel setting. |
| `VLLM_DP_SIZE` | World size of the data parallel setting. |
| `VLLM_DP_MASTER_IP` | IP address of the master node in the data parallel setting. |
| `VLLM_DP_MASTER_PORT` | Port of the master node in the data parallel setting. |
| `VLLM_CI_USE_S3` | Whether to use the S3 path for model loading in CI by using RunAI Streamer. |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | Whether to use atomicAdd reduce in gptq/awq marlin kernel. |
| `VLLM_V0_USE_OUTLINES_CACHE` | Whether to turn on the outlines cache for V0. This cache is unbounded and on disk, so it is unsafe for environments with malicious users. |
| `VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION` | If set, disables TPU-specific optimization for top-k & top-p sampling. |

# Viewing AI Inference Server metrics

vLLM exposes various metrics via the `/metrics` endpoint on the AI Inference Server OpenAI-compatible API server.

You can start the server by using Python, or using [Docker](#deployment-docker).

1. Launch the AI Inference Server server and load your model as shown in the following example. The command also exposes the OpenAI-compatible API.

    ```console
    $ vllm serve unsloth/Llama-3.2-1B-Instruct
    ```
2. Query the `/metrics` endpoint of the OpenAI-compatible API to get the latest metrics from the server:

    ```console
    $ curl http://0.0.0.0:8000/metrics
    ```

    **Example output**

    ```terminal

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    #...
    ```

# AI Inference Server metrics

AI Inference Server exposes vLLM metrics that you can use to monitor the health of the system.

**vLLM metrics**

| Metric Name | Description |
| --- | --- |
| `vllm:num_requests_running` | Number of requests currently running on GPU. |
| `vllm:num_requests_waiting` | Number of requests waiting to be processed. |
| `vllm:lora_requests_info` | Running stats on LoRA requests. |
| `vllm:num_requests_swapped` | Number of requests swapped to CPU. ***Deprecated: KV cache offloading is not used in V1.*** |
| `vllm:gpu_cache_usage_perc` | GPU KV-cache usage. A value of 1 means 100% usage. |
| `vllm:cpu_cache_usage_perc` | CPU KV-cache usage. A value of 1 means 100% usage. ***Deprecated: KV cache offloading is not used in V1.*** |
| `vllm:cpu_prefix_cache_hit_rate` | CPU prefix cache block hit rate. ***Deprecated: KV cache offloading is not used in V1.*** |
| `vllm:gpu_prefix_cache_hit_rate` | GPU prefix cache block hit rate. ***Deprecated: Use `vllm:gpu_prefix_cache_queries` and `vllm:gpu_prefix_cache_hits` in V1.*** |
| `vllm:num_preemptions_total` | Cumulative number of preemptions from the engine. |
| `vllm:prompt_tokens_total` | Total number of prefill tokens processed. |
| `vllm:generation_tokens_total` | Total number of generation tokens processed. |
| `vllm:iteration_tokens_total` | Histogram of the number of tokens per engine step. |
| `vllm:time_to_first_token_seconds` | Histogram of time to the first token in seconds. |
| `vllm:time_per_output_token_seconds` | Histogram of time per output token in seconds. |
| `vllm:e2e_request_latency_seconds` | Histogram of end-to-end request latency in seconds. |
| `vllm:request_queue_time_seconds` | Histogram of time spent in the WAITING phase for a request. |
| `vllm:request_inference_time_seconds` | Histogram of time spent in the RUNNING phase for a request. |
| `vllm:request_prefill_time_seconds` | Histogram of time spent in the PREFILL phase for a request. |
| `vllm:request_decode_time_seconds` | Histogram of time spent in the DECODE phase for a request. |
| `vllm:time_in_queue_requests` | Histogram of time the request spent in the queue in seconds. ***Deprecated: Use `vllm:request_queue_time_seconds` instead.*** |
| `vllm:model_forward_time_milliseconds` | Histogram of time spent in the model forward pass in milliseconds. ***Deprecated: Use prefill/decode/inference time metrics instead.*** |
| `vllm:model_execute_time_milliseconds` | Histogram of time spent in the model execute function in milliseconds. ***Deprecated: Use prefill/decode/inference time metrics instead.*** |
| `vllm:request_prompt_tokens` | Histogram of the number of prefill tokens processed. |
| `vllm:request_generation_tokens` | Histogram of the number of generation tokens processed. |
| `vllm:request_max_num_generation_tokens` | Histogram of the maximum number of requested generation tokens. |
| `vllm:request_params_n` | Histogram of the `n` request parameter. |
| `vllm:request_params_max_tokens` | Histogram of the `max_tokens` request parameter. |
| `vllm:request_success_total` | Count of successfully processed requests. |
| `vllm:spec_decode_draft_acceptance_rate` | Speculative token acceptance rate. |
| `vllm:spec_decode_efficiency` | Speculative decoding system efficiency. |
| `vllm:spec_decode_num_accepted_tokens_total` | Total number of accepted tokens. |
| `vllm:spec_decode_num_draft_tokens_total` | Total number of draft tokens. |
| `vllm:spec_decode_num_emitted_tokens_total` | Total number of emitted tokens. |

# Deprecated metrics

The following metrics are deprecated and will be removed in a future
version of AI Inference Server:

* `vllm:num_requests_swapped`
* `vllm:cpu_cache_usage_perc`
* `vllm:cpu_prefix_cache_hit_rate` (KV cache offloading is not used in V1).
* `vllm:gpu_prefix_cache_hit_rate`. This metric is replaced by queries+hits counters in V1.
* `vllm:time_in_queue_requests`. This metric is duplicated by `vllm:request_queue_time_seconds`.
* `vllm:model_forward_time_milliseconds`
* `vllm:model_execute_time_milliseconds`. Prefill, decode or inference time metrics should be used instead.

<dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

When metrics are deprecated in version `X.Y`, they are hidden in
version `X.Y+1` but can be re-enabled by using the
`--show-hidden-metrics-for-version=X.Y` escape hatch.
Deprecated metrics are completely removed in the following version `X.Y+2`.
</dd></dl>

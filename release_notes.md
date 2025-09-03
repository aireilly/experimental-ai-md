# Release notes

Red Hat AI Inference Server provides developers and IT organizations with a scalable inference platform for deploying and customizing AI models on secure, scalable resources with minimal configuration and resource usage.

## About this release

Red Hat AI Inference Server is now available.
This Red Hat AI Inference Server 3.2.1 release provides container images that optimizes inferencing with large language models (LLMs) for NVIDIA CUDA, AMD ROCm, and Google TPU AI accelerators.
The container images are available from [registry.redhat.io](https://registry.redhat.io):

* `registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1`
* `registry.redhat.io/rhaiis/vllm-rocm-rhel9:3.2.1`
* `registry.redhat.io/rhaiis/vllm-tpu-rhel9:3.2.1`

With Red Hat AI Inference Server, you can serve and inference models with higher performance, lower cost, and enterprise-grade stability and security.
Red Hat AI Inference Server is built on the upstream, open source [vLLM](https://github.com/vllm-project) software project.

## New features and enhancements

Red Hat AI Inference Server 3.2 packages the upstream vLLM v0.10.0 release and includes substantial improvements across model support, performance, and engine architecture.

You can review the complete list of updates in the upstream [vLLM v0.10.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.10.0).

* The Red Hat AI Inference Server supported product and hardware configurations have been expanded.
For more information, see [Supported product and hardware configurations](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.2/html-single/supported_product_and_hardware_configurations/index).

<dl><dt><strong>📌 NOTE</strong></dt><dd>

The Red Hat AI Inference Server 3.2 release does not package LLM Compressor.
</dd></dl>

**AI accelerator performance highlights**

|     |     |     |
| --- | --- | --- |
| Feature | Benefit | Supported GPUs |
| Blackwell compute capability 12.0 | Runs on NVIDIA RTX PRO 6000 Blackwell Server Edition supporting W8A8/FP8 kernels and related tuning | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| ROCm improvements | Full‑graph capture for TritonAttention, quick All‑Reduce, and chunked pre‑fill | AMD ROCm |

### New AI accelerators
Red Hat AI Inference Server 3.2 expands capabilities by enabling you to run workloads on the latest data center grade hardware, now including NVIDIA L20 and Google TPU AI accelerators.

For more information, see [Supported product and hardware configurations](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.2/html-single/supported_product_and_hardware_configurations/index).

### New models enabled
Red Hat AI Inference Server 3.2.1 expands capabilities by enabling the following models:

* Added in vLLM v0.10.0:
    * Llama 4 with EAGLE support
    * EXAONE 4.0
    * Microsoft Phi‑4‑mini‑flash‑reasoning
    * Hunyuan V1 Dense + A13B, including reasoning and tool-parsing abilities
    * Ling mixture-of-experts (MoE) models
    * JinaVL Reranker
    * Nemotron‑Nano‑VL‑8B‑V1
    * Arcee
    * Voxtral
* Added in vLLM v0.9.2:
    * Ernie 4.5
    * MiniMax-M1
    * Slim-MoE
    * Phi‑tiny‑MoE‑instruct
    * Tencent HunYuan‑MoE‑V1
    * Keye‑VL‑8B‑Preview
    * GLM‑4.1 V
    * Gemma‑3
    * Tarsier 2
    * Qwen 3 Embedding & Reranker
    * dots1
    * GPT‑2 for Sequence Classification
* Added in vLLM v0.9.1:
    * Magistral
    * LoRA support for InternVL
    * Minicpm eagle support
    * NemotronH
* Added in vLLM v0.9.0:
    * MiMo-7B
    * MiniMax-VL-01
    * Ovis 1.6, Ovis 2
    * Granite 4
    * FalconH1
    * LlamaGuard4

### New developer features

* **Improved scheduler performance**\
The vLLM scheduler API `CachedRequestData` class has been updated, resulting in improved performance for object and cached sampler‑ID stores.
* **CUDA graph execution**
    * CUDA graph execution is now available for all FlashAttention-3 (FA3) and FlashMLA paths, including prefix‑caching.
    * New live CUDA graph capture progress bar makes debugging easier.
* **Scheduling**
    * Priority scheduling is now implemented in the vLLM V1 engine.
* **Inference engine updates**
    * V0 engine cleanup - removed legacy CPU/XPU/TPU V0 backends.
    * Experimental asynchronous scheduling can be enabled by using the `--async-scheduling` flag to overlap engine core scheduling with the GPU runner for improved inference throughput.
    * Reduced startup time for CUDA graphs by calling `gc.freeze` before capture.
* **Performance improvements**
    * 48% request duration reduction by using micro-batch tokenization for concurrent requests
    * Added fused MLA QKV and strided layernorm.
    * Added Triton causal-conv1d for Mamba models.
* **New quantization options**
    * MXFP4 quantization for Mixture of Experts models.
    * BNB (Bits and Bytes) support for Mixtral models.
    * Hardware-specific quantization improvements.
* **Expanded model support**
    * Llama 4 with EAGLE speculative decoding support.
    * EXAONE 4.0 and Microsoft Phi-4-mini model families.
    * Hunyuan V1 Dense and Ling MoE architectures.
* **OpenAI compatibility**
    * Added new OpenAI Responses API implementation.
    * Added tool calling with required choice and `$defs`.
* **Dependency updates**
    * Red Hat AI Inference Server Google TPU container image uses PyTorch 2.9.0 nightly.
    * NVIDIA CUDA uses PyTorch 2.7.1.
    * AMD ROCm remains on PyTorch 2.7.0.
    * FlashInfer library is updated to v0.2.8rc1.

### Known issues

* In Red Hat AI Inference Server model deployments in OpenShift Container Platform 4.19 with CoreOS 9.6, ROCm driver 6.4.2, and multiple ROCm AI accelerators, model deployment fails.
This issue does not occur with CoreOS 9.4 paired with the matching ROCm driver 6.4.2 version.

    To workaround this ROCm driver issue, ensure that you deploy compatible OpenShift Container Platform and ROCm driver versions:

    **Supported OpenShift Container Platform and ROCm driver versions**

    |     |     |
    | --- | --- |
    | OpenShift Container Platform version | ROCm driver version |
    | 4.17 | 6.4.2 |
    | 4.17 | 6.3.4 |

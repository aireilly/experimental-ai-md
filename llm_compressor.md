# LLM Compressor

LLM Compressor is an open source library that incorporates the latest research in model compression, allowing you to generate compressed models with minimal effort.

The LLM Compressor framework leverages the latest quantization, sparsity, and general compression techniques to improve generative AI model efficiency, scalability, and performance while maintaining accuracy.
With native Hugging Face and vLLM support, you can seamlessly integrate optimized models with deployment pipelines for faster, cost-saving inference at scale, powered by the compressed-tensors model format.

<dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

LLM Compressor is a Developer Preview feature only. Developer Preview features are not supported by Red Hat in any way and are not functionally complete or production-ready. Do not use Developer Preview features for production or business-critical workloads. Developer Preview features provide early access to upcoming product features in advance of their possible inclusion in a Red Hat product offering, enabling customers to test functionality and provide feedback during the development process. These features might not have any documentation, are subject to change or removal at any time, and testing is limited. Red Hat might provide ways to submit feedback on Developer Preview features without an associated SLA.
</dd></dl>

* [LLM Compressor quick start guides](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides)
* [LLM Compressor on GitHub](https://github.com/vllm-project/llm-compressor)

# About large language model optimization

As AI applications mature and new compression algorithms are published, there is a need for unified tools which can apply various compression algorithms that are specific to a users inference needs, optimized to run on accelerated hardware.

Optimizing large language models (LLMs) involves balancing three key factors: model size, inference speed, and accuracy.
Improving any one of these factors can have a negative effect on the other factors.
For example, increasing model accuracy usually requires more parameters, which results in a larger model and potentially slower inference.
The tradeoff between these factors is a core challenge when serving LLMs.

LLM Compressor allows you to perform model optimization techniques such as quantization, sparsity, and compression to reduce memory use, model size, and improve inference without affecting the accuracy of model responses.
The following compression methodologies are supported by LLM Compressor:

* **Quantization**\
Converts model weights and activations to lower-bit formats such as `int8`, reducing memory usage.
* **Sparsity**\
Sets a portion of model weights to zero, often in fixed patterns, allowing for more efficient computation.
* **Compression**\
Shrinks the saved model file size, ideally with minimal impact on performance.

Use these methods together to deploy models more efficiently on resource-limited hardware.

* [Getting started with LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/getting-started/)

# Supported model compression workflows

LLM Compressor supports post-training quantization, a conversion technique that reduces model size and improves CPU and hardware accelerator performance latency, without degrading model accuracy.
A streamlined API applies quantization or sparsity based on a data set that you provide.

The following advanced model types and deployment workflows are supported:

* ***Multimodal models***: Includes vision-language models
* ***Mixture of experts (MoE) models***: Supports models like DeepSeek and Mixtral
* ***Large model support***: Uses the Hugging Face [accelerate](https://github.com/huggingface/accelerate) library for multi-GPU and CPU offloading

All workflows are Hugging Face–compatible, enabling models to be quantized, compressed, and deployed with vLLM for efficient inference.
LLM Compressor supports several compression algorithms:

* ***AWQ***: Weight only `INT4` quantization
* ***GPTQ***: Weight only `INT4` quantization
* ***FP8***: Dynamic per-token quantization
* ***SparseGPT***: Post-training sparsity
* ***SmoothQuant***: Activation quantization

Each of these compression methods computes optimal scales and zero-points for weights and activations.
Optimized scales can be per tensor, channel, group, or token.
The final result is a compressed model saved with all its applied quantization parameters.

* [LLM Compressor examples](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/)
* [Optimize LLMs for low-latency deployments](https://developers.redhat.com/articles/2025/05/09/llm-compressor-optimize-llms-low-latency-deployments)

# Integration with Red Hat AI Inference Server and vLLM

Quantized and sparse models that you create with LLM Compressor are saved using the `compressed-tensors` library (an extension of [Safetensors](https://huggingface.co/docs/safetensors/en/index)).
The compression format matches the model’s quantization or sparsity type.
These formats are natively supported in vLLM, enabling fast inference through optimized deployment kernels by using Red Hat AI Inference Server or other inference providers.

* [Getting started with Red Hat AI Inference Server](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.2/html-single/getting_started/index)

# Integration with Red&#160;Hat OpenShift AI

You can use Red&#160;Hat OpenShift AI and LLM Compressor to experiment with model training, fine-tuning, and compression.
The OpenShift AI integration of LLM Compressor provides two introductory examples:

* A workbench image and notebook that demonstrates the compression of a tiny model, that you can run on CPU, highlighting how calibrated compression can improve over data-free approaches.
* A data science pipeline that extends the same workflow to a larger Llama 3.2 model, highlighting how users can build automated, GPU-accelerated experiments that can be shared with other stakeholders from a single URL.

Both are available in the [Red Hat AI Examples](https://github.com/red-hat-data-services/red-hat-ai-examples/tree/main/examples/llmcompressor) repository.

<dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

The OpenShift AI integration of LLM Compressor is a Developer Preview feature.
</dd></dl>

* [Optimize LLMs with LLM Compressor in Red&#160;Hat OpenShift AI](https://developers.redhat.com/articles/2025/05/20/optimize-llms-llm-compressor-openshift-ai)

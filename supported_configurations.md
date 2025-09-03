# Supported product and hardware configurations

This document describes the supported hardware, software, and delivery platforms that you can use to run Red Hat AI Inference Server in production environments.

<dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

[Technology Preview](https://access.redhat.com/support/offerings/techpreview) and [Developer Preview](https://access.redhat.com/support/offerings/devpreview) features are provided for early access to potential new features.

Technology Preview or Developer Preview features are not supported or recommended for production workloads.
</dd></dl>

* [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/latest)
* [Red Hat AI on Hugging Face](https://huggingface.co/RedHatAI)
* [LLM Compressor techniques](https://developers.redhat.com/articles/2023/03/21/sparsegpt-remove-100-billion-parameters-free)

# Product and version compatibility

The following table lists the supported product versions for Red Hat AI Inference Server 3.2.1.

**Product and version compatibility**

|     |     |
| --- | --- |
| Product | Supported version |
| Red Hat AI Inference Server | 3.2.1 |
| vLLM core | v0.10.0 |
| LLM Compressor | LLM Compressor v0.6.0 is not included in the Red Hat AI Inference Server 3.2.1 container image. |

# Supported AI accelerators

The following tables list the supported AI data center grade accelerators for Red Hat AI Inference Server 3.2.1.

<dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

Red Hat AI Inference Server only supports data center grade accelerators.

Red Hat AI Inference Server 3.2.1 is not compatible with CUDA versions lower than 12.8.
</dd></dl>

**Supported NVIDIA AI accelerators**

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Container image | vLLM release | AI accelerators | Requirements | vLLM architecture support | LLM Compressor support |
| `registry.redhat.io/rhaiis/vllm‑cuda-rhel9:3.2.1` | vLLM v0.10.0 | NVIDIA data center GPUs: * Turing: T4 * Ampere: A2, A10, A16, A30, A40, A100 * Ada Lovelace: L4, L20, L40, L40S * Hopper: H100, H200, GH200 * Blackwell: B200, RTX PRO 6000 Blackwell Server Edition | * [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) * [NVIDIA Container Toolkit 1.14](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.4/release-notes.html) * [NVIDIA GPU Operator 24.3](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.3.0/release-notes.html) * [Python 3.12](https://www.python.org/downloads/release/python-3120/) * [PyTorch 2.7.0](https://docs.pytorch.org/xla/release/r2.7/index.html) | * x86 | Not included by default |

<dl><dt><strong>📌 NOTE</strong></dt><dd>

NVIDIA T4 and A100 accelerators do not support FP8 (W8A8) quantization.
</dd></dl>

**Supported AMD AI accelerators**

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Container image | vLLM release | AI accelerators | Requirements | vLLM architecture support | LLM Compressor support |
| `registry.redhat.io/rhaiis/vllm‑rocm-rhel9:3.2.1` | vLLM v0.10.0 | * AMD Instinct MI210 * AMD Instinct MI300X | * [ROCm 6.2](https://rocm.docs.amd.com/en/docs-6.2.0/about/release-notes.html) * [AMD GPU Operator 6.2](https://rocm.docs.amd.com/en/docs-6.2.4/about/release-notes.html) * [Python 3.12](https://www.python.org/downloads/release/python-3120/) * [PyTorch 2.7.0](https://docs.pytorch.org/xla/release/r2.7/index.html) | x86 | x86 Technology Preview |

<dl><dt><strong>📌 NOTE</strong></dt><dd>

AMD GPUs support FP8 (W8A8) and GGUF quantization schemes only.
</dd></dl>

**Google TPU AI accelerators (Developer Preview)**

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Container image | vLLM release | AI accelerators | Requirements | vLLM architecture support | LLM Compressor support |
| `registry.redhat.io/rhaiis/vllm‑xla-rhel9:3.2.1` | vLLM v0.10.0 | Google v4, v5e, v5p, v6e, Trillium | * [Python 3.12](https://www.python.org/downloads/release/python-3120/) * [PyTorch 2.9.0](https://docs.pytorch.org/xla/) | x86 Developer Preview | Not supported |

# Supported deployment environments

The following deployment environments for Red Hat AI Inference Server are supported.

**Red Hat AI Inference Server supported deployment environments**

|     |     |     |
| --- | --- | --- |
| Environment | Supported versions | Deployment notes |
| OpenShift Container Platform (self‑managed) | 4.14 – 4.19 | Deploy on bare‑metal hosts or virtual machines. |
| Red&#160;Hat OpenShift Service on AWS (ROSA) | 4.14 – 4.19 | Requires a ROSA cluster with STS and GPU‑enabled P5 or G5 node types. See [Prepare your environment](https://docs.redhat.com/en/documentation/red_hat_openshift_service_on_aws/4/html-single/prepare_your_environment/index) for more information. |
| Red&#160;Hat Enterprise Linux (RHEL) | 9.2 – 10.0 | Deploy on bare‑metal hosts or virtual machines. |
| Linux (not RHEL) | - | Supported under third‑party policy deployed on bare‑metal hosts or virtual machines. OpenShift Container Platform Operators are not required. |
| Kubernetes (not OpenShift Container Platform) | - | Supported under third‑party policy deployed on bare‑metal hosts or virtual machines. |

<dl><dt><strong>📌 NOTE</strong></dt><dd>

Red Hat AI Inference Server is available only as a container image.
The host operating system and kernel must support the required accelerator drivers.
For more information, see [Supported AI accelerators](#rhaiis-supported-ai-accelerators_supported-configurations).
</dd></dl>

# OpenShift Container Platform software prerequisites for GPU deployments

The following table lists the OpenShift Container Platform software prerequisites for GPU deployments.

**Software prerequisites for GPU deployments**

|     |     |     |
| --- | --- | --- |
| Component | Minimum version | Operator |
| NVIDIA GPU Operator | 24.3 | [NVIDIA GPU Operator OLM Operator](https://docs.nvidia.com/datacenter/cloud-native/openshift/latest/index.html) |
| AMD GPU Operator | 6.2 | [AMD GPU Operator OLM Operator](https://catalog.redhat.com/software/containers/amd/gpu-operator-v1/675a43ffba424dccbad17635) |
| Node Feature Discovery ^[1]^ | 4.14 | [Node Feature Discovery Operator](https://docs.redhat.com/en/documentation/openshift_container_platform/4.18/html/specialized_hardware_and_driver_enablement/psap-node-feature-discovery-operator) |
[1] Included by default with OpenShift Container Platform.
Node Feature Discovery is required for [scheduling NUMA-aware workloads](https://docs.redhat.com/en/documentation/openshift_container_platform/4.18/html/scalability_and_performance/cnf-numa-aware-scheduling).

## Lifecycle and update policy

Security and critical bug fixes are delivered as container images available from the `registry.access.redhat.com/rhaiis` container registry and are announced through RHSA advisories.
See [RHAIIS container images on catalog.redhat.com](https://catalog.redhat.com/search?gs&q=rhaiis&searchType=containers) for more details.

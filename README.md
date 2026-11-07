## Hi, I'm thisish3

I work on **LLM inference infrastructure**: serving engines, GPU kernels, collective communication, and the systems around them. I like building small versions of real systems, measuring them, and writing down what I find.

[![Blog](https://img.shields.io/badge/Blog-thisish3.tistory.com-EB531F?style=flat-square&logo=tistory&logoColor=white)](https://thisish3.tistory.com)
![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Rust](https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust&logoColor=white)
![Go](https://img.shields.io/badge/Go-00ADD8?style=flat-square&logo=go&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-555555?style=flat-square)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white)

### Building now

| Project | What | Status |
|---|---|---|
| [**hxccl**](https://github.com/Thisish3/hxccl) | An NCCL/RCCL-style collective library over OpenMPI: ring and recursive-doubling AllReduce, the full CPU-side NCCL/RCCL API (all 12 dtypes incl. software fp16/bf16/fp8, groups, Send/Recv, AlltoAll(v), Split/Shrink), benchmarked against OpenMPI. 78 MPI tests | CPU stages done; GPU IPC path next |
| **hxgemm** | FP32 SGEMM step by step (naive → coalescing → shared tiling → register tiling → `cp.async` pipeline) against cuBLAS with Nsight Compute evidence, then Tensor Core, INT8/FP8 and a HIP port | Spec and plan done |
| **ttalkkak** | Pick a Hugging Face model and agent skills, get a measured serving config (vLLM/SGLang) plus agent-readiness checks | Waiting for GPU measurements (private) |

### Open-source contributions

| Project | PR | Status |
|---|---|---|
| llm-d router | [#3828](https://github.com/llm-d/llm-d-router/pull/3828) use the event offset to keep repeated multimodal items in many:1 KV-event realignment | Merged |
| NVIDIA Dynamo frontend | [#477](https://github.com/ai-dynamo/frontend-crates/pull/477) strip a force-reasoning `<think>` that follows text in one chunk | Merged |
| NVIDIA Dynamo frontend | [#481](https://github.com/ai-dynamo/frontend-crates/pull/481) hold back a partial force-reasoning start tag after a prefix | Merged |
| NVIDIA Dynamo frontend | [#482](https://github.com/ai-dynamo/frontend-crates/pull/482) extend the unified chunk-invariance conformance test | Merged |
| NVIDIA Dynamo | [#16399](https://github.com/ai-dynamo/dynamo/pull/16399) pick a complete tool call when `parallel_tool_calls` is false while streaming | Open |
| GuideLLM | [#1546](https://github.com/vllm-project/guidellm/pull/1546) count output tokens from the response text when usage is missing | Open |
| SGLang | [#46295](https://github.com/sgl-project/sglang/pull/46295) minimax-m2: scope streamed parameters to the current invoke | Open |
| SGLang | [#46298](https://github.com/sgl-project/sglang/pull/46298) hunyuan: skip undefined tools in streaming like `detect_and_parse` | Open |
| vLLM | [#56040](https://github.com/vllm-project/vllm/pull/56040) honor `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0` on RDNA | Closed: maintainers preferred `--attention-backend` |
| SGLang | [#39075](https://github.com/sgl-project/sglang/pull/39075) fail clearly when DCP decode hits the LSE-less aiter path | Closed: superseded by AMD's full fix [#34432](https://github.com/sgl-project/sglang/pull/34432) |

### Experiments

**Serving and deployment**
- [vllm-serving-tuning-practice](https://github.com/Thisish3/vllm-serving-tuning-practice): continuous batching throughput, prefix caching TTFT, and decode memory-bandwidth utilization on an RTX 3060 Ti
- [vllm-k8s-deploy-practice](https://github.com/Thisish3/vllm-k8s-deploy-practice): vLLM on Kubernetes with rolling updates, HPA, and a CI latency regression gate, then real vLLM on an RTX 3060 Ti node (k3s on WSL2, throughput within 0.4% of bare Docker over 5 repeated runs)

**Agent workloads**
- [agentic-loop-bottleneck-practice](https://github.com/Thisish3/agentic-loop-bottleneck-practice): real vLLM tool-calling loops; structured decoding reliability, a LangGraph agent profiled with NeMo Agent Toolkit and Nsight Systems, fp16 vs fp8 KV cache
- [harness-cpu-overhead-practice](https://github.com/Thisish3/harness-cpu-overhead-practice): CPU overhead in the agent harness, framed with Amdahl's Law
- [gil-scheduling-practice](https://github.com/Thisish3/gil-scheduling-practice): does LangGraph's parallel node execution use multiple cores under the GIL?

**Kernels, communication, and hardware**
- [vllm-kernel-practice](https://github.com/Thisish3/vllm-kernel-practice): Triton versions of vLLM CUDA kernels
- [allreduce-practice](https://github.com/Thisish3/allreduce-practice): naive, ring, and recursive-doubling allreduce in C with a cost model, plus transfer/compute overlap on a real GPU (grew into hxccl)
- [lpu-compiler-practice](https://github.com/Thisish3/lpu-compiler-practice): toy instruction-block DSL and register allocator modeled on an LPU
- [bloom-filter-ngram-practice](https://github.com/Thisish3/bloom-filter-ngram-practice): Bloom filter n-gram pre-filter modeled on vLLM's n-gram speculative decoding

### Writing

Korean tech blog at [thisish3.tistory.com](https://thisish3.tistory.com): LLM inference optimization (KV cache, speculative decoding, sparse/hybrid attention), serving-engine code reading (vLLM, SGLang), serving verification, and dev logs for the projects above.

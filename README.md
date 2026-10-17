## Hi, I'm Gunhee Lee (이건희)

I work on LLM inference infrastructure: serving engines, kernels, and the systems around them.

### Open-source contributions

- **NVIDIA Dynamo** [#16399](https://github.com/ai-dynamo/dynamo/pull/16399): pick a complete tool call when `parallel_tool_calls` is false while streaming (Rust, Chat Completions and Responses APIs). Under review
- **vLLM** [#56040](https://github.com/vllm-project/vllm/pull/56040): honor an explicit `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0` on AMD RDNA GPUs. Under review
- **SGLang** [#39075](https://github.com/sgl-project/sglang/pull/39075): traced a decode context parallel crash to the aiter decode path returning no LSE; closed after AMD landed the full fix in [#34432](https://github.com/sgl-project/sglang/pull/34432)

### Projects

**Serving and deployment**
- [vllm-serving-tuning-practice](https://github.com/Thisish3/vllm-serving-tuning-practice): continuous batching throughput and prefix caching TTFT benchmarks
- [vllm-k8s-deploy-practice](https://github.com/Thisish3/vllm-k8s-deploy-practice): vLLM on Kubernetes with rolling updates, health checks, HPA, and a CI latency regression gate

**Agent workloads**
- [agentic-loop-bottleneck-practice](https://github.com/Thisish3/agentic-loop-bottleneck-practice): real vLLM tool-calling loop; structured decoding reliability and LLM vs tool-execution time
- [harness-cpu-overhead-practice](https://github.com/Thisish3/harness-cpu-overhead-practice): CPU overhead in the agent harness, framed with Amdahl's Law
- [gil-scheduling-practice](https://github.com/Thisish3/gil-scheduling-practice): does LangGraph's parallel node execution use multiple cores under the GIL?

**Kernels, communication, and hardware**
- [vllm-kernel-practice](https://github.com/Thisish3/vllm-kernel-practice): Triton versions of vLLM CUDA kernels
- [allreduce-practice](https://github.com/Thisish3/allreduce-practice): naive, ring, and recursive-doubling allreduce in C with a cost model
- [lpu-compiler-practice](https://github.com/Thisish3/lpu-compiler-practice): toy instruction-block DSL and register allocator modeled on an LPU
- [bloom-filter-ngram-practice](https://github.com/Thisish3/bloom-filter-ngram-practice): Bloom filter n-gram pre-filter modeled on vLLM's n-gram speculative decoding

### Writing

Blog: [thisish3.tistory.com](https://thisish3.tistory.com)

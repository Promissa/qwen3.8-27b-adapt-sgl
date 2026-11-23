# Operational checks

- A user terminal and an agent sandbox can see different devices/network access.
  Check `/dev/dxg`, `nvidia-smi`, `torch.cuda.is_available()`, and local socket
  availability in the exact execution session. Approved execution outside the sandbox has working GPU, CUDA and
  socket access; sandboxed execution is restricted.
- NVFP4 package import is not a successful FP4 GEMM. First execution may require
  CUDA headers/nvcc and JIT resources. Runtime setup supplies a matching local CUDA 13.0 compiler/header overlay.
- Keep the 0.5.21 release and source@c3c292b65a results separate. Reuse mode reads
  another environment and source checkout; it is not an immutable wheel install.
- Do not change the existing tokenizer-manager patch. The isolated checkout has
  the Qwen3.5 text-only allowlist and embedding-dispatch fixes; source-dependent
  performance switches remain upstream features.
- Windows sysmem fallback can invalidate memory sweeps. Its setting and moving
  desktop output are user-side prerequisites, not inferred from throughput alone.
  The fresh NVAPI recheck still cannot resolve the known fallback setting; the
  control-panel package/current-user registration and standard legacy executable
  checks found no client. NVIDIA still drives the physical display. These bounded
  checks leave policy verification and the no-desktop condition open; they do not
  establish that every possible settings interface is absent.
- FP8 KV dtype is explicit. `auto` can choose BF16, doubling KV storage. NVFP4 KV
  has a separate decode backend and separate lossy quality gate.
- The main target has no bundled MTP. EAGLE therefore needs an explicit independent
  MTP draft or a target
  checkpoint that includes MTP. Do not substitute the original BF16 DFlash draft
  for the separate quantized candidate without recording that change.
  Opt-in patch0004 permits the converted external MTP to pass24576-token capacity;
  strict greedy still matches0/8, so it is not an admitted lossless recipe.
- Launch success alone is not an envelope pass: prefill activation OOM is common
  near the memory boundary. A server crash, HTTP failure or truncated generation
  is a failed point. Infrastructure failures must stop, not shrink, a memory search.
- `nvidia-smi` samples are whole-device measurements at finite intervals. Small
  activation spikes and allocator-only peaks require profiler/device instrumentation.
- Old `conc_ladder.py` assumes `vllm:*` metrics and substitutes zero for missing
  series. The adapted SGLang ladder retains missing metrics as unknown and every
  repeat; only live endpoint evidence can validate its runtime measurements.
- Never fill measured result tables with the estimates in PLAN.md or third-party
  32GB results. Empty/unmeasured entries are intentional until live experiments.
- In this commit, `profile_by_stage=true` needs a positive `num_steps`; omitting
  it leaves stage counters as None and crashes the scheduler on the next batch.
  The capture driver uses 32 steps and checks for actual decode GPU trace events.
- An unused loopback-port connect can wait through WSL TCP retries. Probe port
  preflight uses a reusable bind instead; it never connects to Windows to decide
  whether a Linux listener can be launched.
- Native control endpoints can acknowledge in plain text; generation/info
  responses remain strict JSON. Do not classify a client parsing error as OOM.

- The Windows NVAPI fallback-policy update returned SETTING_NOT_FOUND (-160),
  before SaveSettings. The policy remains unverified. A successful serving point
  remains a tested workload, but cannot certify a physical-VRAM OOM boundary.
  A direct private-ID SetSetting attempt in a disposable, unsaved session also
  returned−160; fresh-session reads remained unchanged. Public-name enumeration
  alone was insufficient to establish that direct-setting failure.

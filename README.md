vLLM Installation Guide for Proxmox LXC with GPU Passthrough(Placeholder for vLLM Architecture Infographic)1. Introduction: What is vLLM?If you are new to AI hosting, you might be wondering why you need vLLM instead of just running a model directly.In simple terms: vLLM is a high-speed engine that allows your computer (specifically your GPU) to serve AI models to many users at once without running out of memory.The "Bookshelf" Analogy (PagedAttention)Traditional AI engines work like a disorganized library. When a user asks a question, the engine reserves a huge, empty bookshelf just for that conversation, even if the answer is only one sentence long. This wastes a massive amount of space (VRAM).vLLM uses something called PagedAttention. It acts like a smart librarian who tears the pages out of the book and stuffs them into any tiny open slot available on any shelf. It keeps a "map" of where every page is.Result: No empty shelf space is wasted.Benefit: You can fit 2x to 4x more "conversations" (requests) into the same GPU memory.Why use vLLM?Speed: It generates text significantly faster than standard HuggingFace transformers.Efficiency: It manages memory like an Operating System, preventing "Out of Memory" errors when batching requests.Production Ready: It is designed to run as a server (API) that looks exactly like OpenAI's API.2. vLLM vs. llama.cpp: Which one do I need?Both are excellent, but they serve different purposes. Use this table to decide:FeaturevLLMllama.cppBest ForHigh-Performance Serving.Ideal if you want to host an API for multiple users or agents.Personal/Local Use.Ideal for running a model on a laptop or single consumer PC.HardwareRequires NVIDIA GPUs (mostly) with good VRAM. Optimizes for pure speed.Runs on Anything (CPU, Apple Silicon, AMD, NVIDIA). Optimizes for compatibility.Model FormatUses uncompressed weights (BF16/FP16) or GPTQ/AWQ.Uses GGUF (heavily compressed/quantized) models.ThroughputExtremely High.Can handle many requests at the exact same time.Lower.Processes requests sequentially or with lower concurrency.Memory TechPagedAttention.Sophisticated memory management for batching.mmap.Simple memory mapping, low overhead but less efficient at scale.The VerdictUse this if building a Home Lab Server or Production App.Use this if running a chatbot locally on your MacBook or Gaming PC.3. Prerequisites & Critical Resources[!IMPORTANT]The CUDA toolkit must be installed on BOTH the Proxmox host AND the LXC container. This is commonly missed and causes flash-attn compilation failures (segfaults).Companion GuideFor detailed instructions on configuring the LXC container, installing NVIDIA drivers, and managing the passthrough process, please refer to our dedicated guide:👉 GPU Passthrough for Proxmox LXC Container (GitHub)[!NOTE]Use the guide above as the primary reference for Phase 1 & 2 if you encounter specific Proxmox issues.System RequirementsComponentValueProxmox VE8.4+LXC OSUbuntu 22.04GPUNVIDIA RTX 3080 (10GB VRAM) or betterNVIDIA Driver580.x (Must act on Host AND LXC)CUDA Toolkit12.8 (Must match on Host AND LXC)Python3.10.x4. Installation StepsPhase 0: Prepare the Proxmox HostThese steps must be performed on the Proxmox Node (Shell).Phase 0A: Blacklist Nouveau DriversEnsure the open-source drivers do not conflict with Nvidia.echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf
echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist.conf
update-initramfs -u
reboot
Phase 0B: Install Prerequisites on HostWe need a comprehensive set of build tools and libraries to ensure the NVIDIA drivers and CUDA toolkit compile and run correctly.apt install -y \
    g++ \
    freeglut3-dev \
    build-essential \
    libx11-dev \
    libxmu-dev \
    libxi-dev \
    libglu1-mesa-dev \
    libfreeimage-dev \
    libglfw3-dev \
    wget \
    htop \
    btop \
    nvtop \
    glances \
    git \
    pciutils \
    cmake \
    curl \
    libcurl4-openssl-dev \
    pve-headers-$(uname -r)
Phase 0C: Install CUDA Toolkit 12.8 on HostWe will use the official NVIDIA repository to install the CUDA toolkit.wget [https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb](https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb)
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-toolkit-12-8
Phase 0D: Configure CUDA PATH on HostThis step is vital for the host to expose the correct CUDA version.# Add to .bashrc
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi
Phase 1: Create LXC ContainerCreate a privileged Ubuntu 22.04 container.Uncheck "Unprivileged container" (Must be privileged).Resources: Assign at least 32GB Disk, 16GB RAM, 4-8 Cores.Phase 2: Configure Passthrough (Host Side)Refer to the Companion Guide for deep-dive details.Edit /etc/pve/lxc/YOUR-ID.conf on the host. Add the following (Verify device numbers 195 and 508 with ls -l /dev/nvidia*):# GPU Passthrough
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.cgroup2.devices.allow: c 508:* rwm
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-modeset dev/nvidia-modeset none bind,optional,create=file
lxc.mount.entry: /usr/local/cuda-12.8 usr/local/cuda-12.8 none bind,ro,create=dir
Note: The last line binds the CUDA toolkit from Host to LXC, saving space and ensuring version matching.Restart the LXC container now.Phase 3: Setup LXC Environment & Drivers[!CAUTION]CHECK DRIVERS FIRSTExpectation: Both the Proxmox Host AND the LXC Container must have NVIDIA drivers installed and recognized (nvidia-smi) before you proceed to Phase 4.If nvidia-smi is NOT working on your Host: STOP. Fix your host drivers first.If nvidia-smi is NOT working on your LXC (after performing the steps below): STOP.Solution: If you are unable to get nvidia-smi working, please follow the GPU Passthrough for Proxmox LXC Container Guide BEFORE moving forward with this guide.Access the container shell.Phase 3A: Install System Prerequisites (LXC)We must install the same build tools inside the LXC as we did on the host to ensure flash-attn can compile using the mounted CUDA toolkit.apt update && apt install -y \
    g++ \
    freeglut3-dev \
    build-essential \
    libx11-dev \
    libxmu-dev \
    libxi-dev \
    libglu1-mesa-dev \
    libfreeimage-dev \
    libglfw3-dev \
    wget \
    htop \
    btop \
    nvtop \
    glances \
    git \
    pciutils \
    cmake \
    curl \
    libcurl4-openssl-dev \
    python3-dev \
    python3-venv
Phase 3B: Install NVIDIA Driver (LXC)Copy the same .run file from the host to the LXC.sh NVIDIA-Linux-x86_64-*.run --no-kernel-modules
Install OpenGL? NoRun nvidia-xconfig? NoPhase 3C: Configure CUDA PATH (LXC)Even though we mounted the CUDA folder, the LXC doesn't know where it is yet. We must add it to the path.# Add to .bashrc inside LXC
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify (Critical)
nvidia-smi   # Should show GPU
nvcc --version # Should show CUDA 12.8
Phase 4: Prepare Python EnvironmentWe use python venv to avoid conflicts.mkdir -p /opt/vllm
cd /opt/vllm

python3 -m venv venv
source venv/bin/activate

# Upgrade pip (Fixes hash errors with pip 25.x)
pip install --upgrade pip
Phase 5: Install vLLMpip install vllm
Phase 6: Install Flash AttentionRequired for PagedAttention performance.pip install packaging
# This compiles from source and may take 10-20 minutes
pip install flash-attn --no-build-isolation
[!TIP]If compilation fails due to RAM, use export MAX_JOBS=1 before installing.Phase 7: HuggingFace Authentication (Optional)If you plan to use gated models (like Llama 3 or Mistral), you must authenticate.pip install huggingface_hub
huggingface-cli login
# Paste your HF Token when prompted
Phase 8: Verify Installation (Test Script)We will generate a python script to verify that vLLM can access the GPU and that Flash Attention is working.cat << 'SCRIPT' > vllm_test_flash.py
from vllm import LLM, SamplingParams
import torch

# Check CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# Sample Prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize vLLM (This checks PagedAttention and FlashAttn)
llm = LLM(model="facebook/opt-125m")

# Generate
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
SCRIPT
Run the test:python vllm_test_flash.py
Phase 9: Create Startup Scriptcat << 'EOF' > start_vllm.sh
#!/bin/bash
source /opt/vllm/venv/bin/activate
vllm serve "facebook/opt-125m" --host 0.0.0.0 --port 8000
EOF

chmod +x start_vllm.sh
5. References & Further ReadingThis guide was compiled using technical documentation and research from the following sources:Core Research & Documentation:vLLM Official DocumentationPaper: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., arXiv) - Read PaperRunPod Blog: "Introduction to vLLM and PagedAttention" - Read ArticleComparisons & Architecture:Northflank Blog: "vLLM vs Ollama: Key differences, performance, and how to run them"Arm Learning Paths: "Explore llama.cpp architecture and the inference workflow"Red Hat: "What is vLLM?"Community Guides:GPU Passthrough: GitHub - en4ble1337/GPU-Passthrough-for-Proxmox-LXC-Container

### 📂 Recommended Repository Name: `vllm-proxmox-lxc-guide`

---

```markdown
# vLLM Installation Guide for Proxmox LXC with GPU Passthrough (v11)

![vLLM Setup Infographic](infographic.png)

## 📖 Introduction: What is vLLM?

If you are just starting your AI journey, you might wonder why everyone is talking about **vLLM**.

In simple terms, **vLLM is a high-performance engine that makes Large Language Models (LLMs) run faster and more efficiently on your GPU.**

When standard AI programs run, they often waste a lot of GPU memory because they reserve space they don't actually use. vLLM uses a smart technology called **PagedAttention** (inspired by how operating systems manage computer RAM). It breaks memory into small, flexible "blocks" that can be moved around instantly.

### Why should you use it?
* **Speed (High Throughput):** It can handle many more requests per second than standard methods.
* **Efficiency:** It reduces memory waste, allowing you to run larger models on the same hardware.
* **Industry Standard:** It is the engine behind many top serving platforms (like RunPod, Modal, and others).

### 🆚 vLLM vs. llama.cpp
Choosing the right engine depends on your hardware and goals. Here is a quick comparison:

| Feature | **vLLM** | **llama.cpp** |
| :--- | :--- | :--- |
| **Best For** | **High-performance GPU Servers** (NVIDIA/AMD) | **Consumer Hardware** (MacBooks, CPUs, older GPUs) |
| **Primary Speed** | Extremely fast "Throughput" (serving many users) | Low "Latency" (fast response for 1 user) |
| **Model Format** | Unquantized, AWQ, GPTQ, SqueezeLLM | **GGUF** (Highly compressed) |
| **Memory Mgmt** | **PagedAttention** (GPU Memory optimization) | Standard Loading (Optimized for low VRAM/RAM) |
| **Setup Difficulty** | Intermediate (Requires Linux/Drivers) | Easy (Often just one binary file) |
| **Use Case** | Production APIs, Multi-user Chatbots, Enterprise | Local Personal Assistant, Edge Devices, Offline |

---

## 🛠️ Environment & Prerequisites

### Environment Reference
| Component | Value |
| :--- | :--- |
| **Proxmox VE** | 8.4+ (Debian 12 Base) |
| **LXC OS** | Ubuntu 22.04 |
| **GPU** | NVIDIA RTX 30xx/40xx/50xx |
| **NVIDIA Driver** | 580.x (Must match on Host & LXC) |
| **CUDA Toolkit** | 12.8 (Must match on Host & LXC) |
| **Python** | 3.10.x |

### 🔗 Critical Prerequisite Guides
You must perform these steps **before** installing vLLM.

1.  **GPU Passthrough:** The LXC container must see the GPU.
    * 👉 **[Guide: GPU Passthrough for Proxmox LXC](https://github.com/en4ble1337/GPU-Passthrough-for-Proxmox-LXC-Container)**
2.  **BIOS Settings:** Ensure "Above 4G Decoding" and "Re-Size BAR" are enabled in your host BIOS.

---

## PART 1: PROXMOX HOST SETUP

**⚠️ CRITICAL:** Do not skip this section! Flash-attn will segfault without host CUDA. These steps are performed on the **Proxmox host**, not inside an LXC.

Access via: `Proxmox web UI` → `Select node` → `Shell` (or SSH to host).

### Phase 0A: Install NVIDIA Driver on Host (Manual Method)
*Skip if already installed. Verify with `nvidia-smi`.*

1.  **Download Driver (580.x or latest):**
    ```bash
    wget [https://us.download.nvidia.com/XFree86/Linux-x86_64/580.76.05/NVIDIA-Linux-x86_64-580.76.05.run](https://us.download.nvidia.com/XFree86/Linux-x86_64/580.76.05/NVIDIA-Linux-x86_64-580.76.05.run)
    chmod +x NVIDIA-Linux-x86_64-580.76.05.run
    ```
2.  **Install Prerequisites:**
    ```bash
    apt update && apt install build-essential pve-headers-$(uname -r) pkg-config -y
    ```
3.  **Blacklist Nouveau:**
    ```bash
    echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf
    echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
    update-initramfs -u
    reboot
    ```
4.  **Install Driver:**
    * *Note: For RTX 50xx (Blackwell), select "MIT" drivers if prompted.*
    ```bash
    ./NVIDIA-Linux-x86_64-580.76.05.run --dkms
    ```

### Phase 0B: Install Prerequisites on Host
Ensure standard build tools are present for the CUDA installer.
```bash
apt install build-essential linux-headers-$(uname -r) -y

```

### Phase 0C: Install CUDA Toolkit 12.8 on Host

*Note: We install the toolkit **without** the driver to prevent overwriting your manual driver installation.*

```bash
wget [https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb](https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb)
dpkg -i cuda-repo-debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb
cp /var/cuda-repo-debian12-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt update
apt install cuda-toolkit-12-8 -y

```

### Phase 0D: Configure CUDA PATH on Host

Add CUDA to your path so `nvcc` works.

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

```

### Phase 0E: Reboot Host and Verify

Reboot the host to ensure all modules are loaded.

```bash
reboot

```

**Verification:**

* `nvidia-smi` should show Driver 580.x and CUDA 12.8.
* `nvcc --version` should show "release 12.8".
* *If `nvidia-smi` fails:* Re-run the `.run` driver installer from Phase 0A.

---

## PART 2: LXC CONTAINER SETUP

**These steps are performed inside the LXC container.**

### Phase 1: Create LXC with GPU Passthrough

Create an Ubuntu 22.04 LXC using the [External Passthrough Guide](https://github.com/en4ble1337/GPU-Passthrough-for-Proxmox-LXC-Container).

* **Resources:** 4+ Cores, 8GB+ RAM, 50GB Disk.

### Phase 2: Install NVIDIA Driver in LXC

We must install the **exact same driver version** as the host, but without the kernel modules.

1. **From Proxmox Host:** Push the installer file to the LXC (replace `105` with your LXC ID).
```bash
pct push 105 NVIDIA-Linux-x86_64-580.76.05.run /root/NVIDIA-Linux-x86_64-580.76.05.run

```


2. **Inside LXC:** Install without kernel modules.
```bash
chmod +x NVIDIA-Linux-x86_64-580.76.05.run
./NVIDIA-Linux-x86_64-580.76.05.run --no-kernel-modules

```



### Phase 3: Install Prerequisites in LXC

```bash
apt update && apt upgrade -y
apt install build-essential git curl wget software-properties-common python3-pip python3-venv python3-dev -y

```

### Phase 4: Install CUDA Toolkit 12.8 in LXC

*Same as host, but for Ubuntu 22.04.*

```bash
wget [https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin)
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget [https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb](https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb)
dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt update
apt install cuda-toolkit-12-8 -y

```

### Phase 5: Configure CUDA PATH in LXC

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

```

### Phase 6: Create Python Environment

We use a virtual environment to avoid conflicts.

```bash
mkdir -p ~/vllm-project
cd ~/vllm-project
python3 -m venv .venv
source .venv/bin/activate
# Update pip
pip install --upgrade pip setuptools wheel

```

### Phase 7: Install vLLM and Dependencies

```bash
pip install vllm
# Optional: Install flash-attn manually if pre-built wheels fail, 
# but vLLM usually includes 'flashinfer' or its own backend.

```

### Phase 8: Configure HuggingFace

Required to download models like Llama 3 or Mistral.

```bash
pip install huggingface-hub
huggingface-cli login
# Paste your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```

### Phase 9: Set Environment Variables

Optimize for your GPU architecture.

```bash
# Example for RTX 30xx/40xx (Ampere/Ada)
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

```

---

### Phase 10: 🧪 Verify Flash-Attention (Updated)

We will use a comprehensive Python script to verify that vLLM is installed and using an optimized backend (Flash Attention or FlashInfer).

**Copy and paste this entire block into your terminal:**

```bash
cat << 'SCRIPT' > vllm_test_flash.py
import sys
import os

# Set environment to force usage if needed, though vLLM usually detects valid libs
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

try:
    from vllm import LLM, SamplingParams
    import torch
except ImportError:
    print("❌ Critical: vLLM or torch not installed.")
    sys.exit(1)

print("="*50)
print("⚡ vLLM FLASH ATTENTION DIAGNOSTIC ⚡")
print("="*50)

# 1. CHECK DEPENDENCIES
print("\n[1] Checking Acceleration Libraries:")
drivers = []

try:
    import flash_attn
    print(f"  ✓ flash-attn found: {flash_attn.__version__}")
    drivers.append("flash-attn")
except ImportError:
    print("  - flash-attn not installed")

try:
    import flashinfer
    print(f"  ✓ flashinfer found: {flashinfer.__version__}")
    drivers.append("flashinfer")
except ImportError:
    print("  - flashinfer not installed")

if not drivers:
    print("  ⚠️  WARNING: No optimized backend found. vLLM will be slow.")

# 2. FUNCTIONAL TEST
print("\n[2] Initializing vLLM with 'TinyLlama'...")
# Use the main guard to prevent multiprocessing recursive loop issues
if __name__ == "__main__":
    try:
        # We disable logging stats to keep output clean
        llm = LLM(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            gpu_memory_utilization=0.6,
            dtype="float16",
            disable_log_stats=True
        )
        
        print("\n[3] Generating test token...")
        sampling_params = SamplingParams(temperature=0, max_tokens=5)
        output = llm.generate(["Test"], sampling_params)
        
        print("\n" + "="*50)
        print("RESULT:")
        if output and len(drivers) > 0:
            print(f"✅ SUCCESS: vLLM is generating using {drivers[0]}.")
        elif output:
            print("⚠️  PARTIAL SUCCESS: Generation works, but using unoptimized backend.")
        else:
            print("❌ FAILURE: Generation failed.")
        print("="*50)

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        sys.exit(1)
SCRIPT

python3 vllm_test_flash.py

```

---

### Phase 11: Start API Server

Once verified, start the server accessible to your network.

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8000 \
  --host 0.0.0.0

```

### Phase 12: Create Snapshot

**Important:** In Proxmox, create a snapshot now.
`Datacenter` → `Your LXC` → `Snapshots` → `Take Snapshot` (Name: "vLLM-Installed").

---

## 🛠️ Troubleshooting

| Problem | Root Cause | Solution |
| --- | --- | --- |
| **flash-attn segfault** | CUDA toolkit missing on Proxmox host | **Critical:** Install CUDA on Host AND LXC (Part 1). |
| **nvidia-smi error** | CUDA toolkit overwrites driver files | Re-run `./NVIDIA...run` installer on Host. |
| **pipenv ImportError** | Broken system pipenv on Ubuntu | Use `python3 -m venv` (Phase 6). |
| **OOM compilation** | Not enough RAM for build | Use `export MAX_JOBS=1` before installing. |

---

## 📚 References

* **Official Documentation:** [vLLM Docs](https://docs.vllm.ai/en/latest/)
* **The Paper:** [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al.)](https://arxiv.org/abs/2309.06180)
* **vLLM vs llama.cpp Analysis:** [Wallaroo.ai Analysis](https://www.wallaroo.ai/)
* **Proxmox Passthrough:** [En4ble1337's LXC Guide](https://github.com/en4ble1337/GPU-Passthrough-for-Proxmox-LXC-Container)

```

For further visual guidance, you might find this walkthrough helpful:
[Setup vLLM Local Ai in Proxmox 9 LXC](https://www.youtube.com/watch?v=APBpBFZIVEk)

This video covers the specific combination of Proxmox 9, LXC containers, and vLLM installation that matches your guide's architecture.


http://googleusercontent.com/youtube_content/0

```

# NVIDIA GPU + Docker setup (WSL2 and native Linux)

How to get `docker run --gpus all` working so the training image can see your NVIDIA
GPU. This fixes the common error:

```
could not select device driver "nvidia" with capabilities: [[gpu]]
```

which means Docker can't find the NVIDIA container runtime — the **NVIDIA Container
Toolkit** is not installed or not registered with the Docker daemon. The container never
starts; nothing in this project causes it.

> This project's `torch` (2.13.0) bundles the **CUDA 13** runtime. You do **not** install a
> CUDA toolkit on the host — only the **driver** (which provides `libcuda`/`nvidia-smi`)
> and the **container toolkit**. The CUDA userspace libraries ship inside the pip wheel.

---

## Part A — WSL2 (Ubuntu on Windows)

### 1. Install the NVIDIA driver on **Windows** (not inside WSL)

Install the standard Windows NVIDIA driver (GeForce / Studio / Data Center). Recent
drivers include WSL2 GPU support. **Do not install a Linux GPU driver inside the WSL
distro** — it breaks WSL's GPU passthrough.

Make sure you are on **WSL 2**, not WSL 1:

```powershell
# in Windows PowerShell
wsl --status         # should show default version 2
wsl --update         # keep the WSL kernel current
```

### 2. Verify the GPU reaches WSL

In the Ubuntu WSL shell:

```bash
nvidia-smi
```

- **Shows your GPU** → driver + passthrough are fine. The reported **CUDA Version**
  (top-right) is the max your driver supports; it must be **≥ 13.0** for this project.
  Continue to step 3.
- **Fails / not found** → the GPU isn't reaching WSL. Update the **Windows** driver and
  run `wsl --update`, then re-check. Don't proceed until `nvidia-smi` works here.

### 3. Which Docker are you running?

```bash
docker context ls        # a `desktop-linux` context => Docker Desktop
which docker
```

- **Docker Desktop (Windows) with WSL integration** → you do **not** run `nvidia-ctk`.
  Update to a recent Docker Desktop, ensure the Windows driver is installed, then
  **restart Docker Desktop**. GPU support is built in. If it still fails, toggle WSL
  integration off/on in Docker Desktop → Settings → Resources → WSL integration.
  Skip to [Verify](#verify).
- **Docker Engine installed inside the WSL distro** → continue to step 4.

### 4. Install the NVIDIA Container Toolkit (inside WSL)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 5. Register the runtime with Docker and restart

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart          # or: sudo systemctl restart docker
```

This step is what actually clears the `could not select device driver "nvidia"` error.

---

## Part B — Native Linux (Ubuntu / Debian)

### 1. Install the NVIDIA driver

```bash
# easiest on Ubuntu: let the tool pick a recommended driver
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, confirm:

```bash
nvidia-smi           # must show the GPU; CUDA Version (top-right) must be >= 13.0
```

If your distro doesn't have `ubuntu-drivers`, install a specific driver package
(`sudo apt-get install -y nvidia-driver-<version>`) or follow NVIDIA's driver docs.

### 2. Install the NVIDIA Container Toolkit

Same commands as WSL step 4:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 3. Register the runtime and restart Docker

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Verify

A minimal GPU container (uses only the driver, so any CUDA base image works):

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

> The `12.4` here is irrelevant — `nvidia-smi` reports the host **driver**, not the
> image's toolkit, so this only proves `--gpus all` passthrough works.

The truer, project-specific check — the container sees the GPU through **this project's
CUDA 13 torch** (run from the repo root after building the image):

```bash
docker compose -f deploy/docker-compose.yml build
docker run --rm --gpus all --entrypoint python resnet18-cifar10:latest \
  -c "import torch; print('cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

Expect `cuda available: True` with your GPU name. Then train:

```bash
docker compose -f deploy/docker-compose.yml up
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | Container toolkit not installed / not registered with Docker | Part A step 4–5 (or restart Docker Desktop) |
| `nvidia-smi` not found **in WSL** | Windows driver missing/old, or WSL 1 | Install/update Windows driver; `wsl --update`; ensure WSL 2 |
| `nvidia-smi` works but `--gpus all` fails | Docker daemon not restarted after `nvidia-ctk configure` | `sudo service docker restart` |
| `torch.cuda.is_available()` is `False` inside container, but `nvidia-smi` works | Driver too old for CUDA 13, or CPU-only wheel resolved | Update driver so `nvidia-smi` CUDA Version ≥ 13.0; ensure image built on x86_64 |
| `DataLoader worker (pid ...) is killed by signal: Bus error` | `/dev/shm` too small in container | already handled: `shm_size: "2gb"` in compose |

### References

- [NVIDIA Container Toolkit — install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [CUDA on WSL — user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

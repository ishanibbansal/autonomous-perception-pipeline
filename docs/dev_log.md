# Engineering Log & Scratchpad

This file serves as a living technical journal for the project. It tracks active debugging sessions, environment quirks, and architectural decisions. Capturing these roadblocks and solutions builds a concrete knowledge base for technical deep-dives and portfolio reviews.

---

## 🛠️ Environment & Infrastructure Logs (Sprint 0)

### Log 0.1: Remote SSH Terminal Missing `nvidia-smi` Path
* **Date:** July 17, 2026
* **Symptom:** Running `nvidia-smi` locally in WSL2 works perfectly, but executing it over a remote VS Code SSH session throws: `bash: nvidia-smi: command not found`.
* **Root Cause:** WSL2 automatically injects Windows host paths (including the folder where the virtualized NVIDIA drivers live) during standard local logins. However, remote non-interactive or standard SSH sessions skip this injection, leaving the shell blind to `/usr/lib/wsl/lib`.
* **Solution:** Manually append the WSL driver directory to the system path inside the user configuration:
  ```bash
  echo 'export PATH=$PATH:/usr/lib/wsl/lib' >> ~/.bashrc
  source ~/.bashrc
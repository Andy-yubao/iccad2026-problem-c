# Problem C — Submission Guidelines

> 来源：[官方 PDF](C_Submission_Guidelines_20260616.pdf)。正文仅整理换行。
> 阅读前注意：CUDA 12.5 已被 [Q&A A22](official_qa.md#a22) 更正为 13.0；硬件内存规格有多处不同表述，见[研究导读](../../docs/research_guide.zh-CN.md)。

```text
ICCAD 2026 CAD Contest
The Floorset Challenge (Problem C)
Submission Guidelines
Please submit a single compressed (.tar.gz) folder containing the following:
1. Optimizer executable (PyInstaller recommended for Python submissions)
2. requirements.txt (if the executable does not bundle all dependencies)
3. External files such as model checkpoints, configuration files, and helper scripts
4. Optional README file with any instructions needed to run your code
As a fallback, you may also submit your source code. No docker images or files allowed.
System Specifications
Operating System: Debian GNU/Linux 13
Python Environment: Python 3.13.14
PyTorch: 2.12.0+cu130
CUDA: 12.5
GCC: 14.2.0
G++: 14.2.0
GLIBC (ldd): 2.41
Machine Specifications
CPU: Intel Xeon, up to 48 cores, 250 GB RAM (multi-threading allowed)
GPU: NVIDIA A100, 80 GB VRAM
Runtime: No upper limits (runtime measured only during calls to solve() method)
Internet access: No
Evaluation: We will run your optimizer on the hidden test cases during the evaluation in the
same command format provided in the evaluation script.
```

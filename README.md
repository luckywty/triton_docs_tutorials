# Triton 教程环境

本项目使用 uv 管理 `.venv`，依赖通过 `uv.lock` 锁定，使用阿里云 PyPI 镜像。

- Python 3.12
- PyTorch 2.9.1（Linux x86_64 CUDA 12.8 版本）
- Triton 3.5.1（与 PyTorch 的依赖要求一致）
- NumPy

## 安装与验证

```bash
cd /home/wangwei/ai-infra/triton_docs_tutorials
uv sync --locked
uv run verify_env.py
```

验证脚本检查以下导入，并在 GPU 上编译、执行 Triton 向量加法，与 PyTorch 结果比较：

```python
import torch
import triton
import triton.language as tl
```

## 运行教程

可以直接使用 uv：

```bash
uv run python 01_vector_addtion/1.py
```

也可以激活环境后运行：

```bash
source .venv/bin/activate
python 01_vector_addtion/1.py
deactivate
```

在编辑器中，将 Python 解释器设为本目录下的 `.venv/bin/python`。

## 使用 Jupyter

在项目目录启动 JupyterLab：

```bash
uv run jupyter lab
```

也可以启动 Notebook 界面：

```bash
uv run jupyter notebook
```

在 Jupyter 或 VS Code 的 Notebook 中选择内核 `Python (triton_docs_tutorials)`。
该内核使用本项目的 `.venv/bin/python`，可以访问已安装的 PyTorch、Triton 和 CUDA。

首次在新机器上配置时，在 `uv sync --locked` 后注册内核：

```bash
uv run python -m ipykernel install --user --name triton-docs-tutorials --display-name "Python (triton_docs_tutorials)"
```

## Notebook 图片

Jupyter 笔记中使用的教学图片统一存放在项目根目录的 `images/` 文件夹，文件名建议以教程编号开头，例如 `05-layernorm-parallel-reduction.png`。

各教程子目录中的 Notebook 使用相对路径引用图片：

```markdown
![两阶段并行归约](../images/05-layernorm-parallel-reduction.png)
```

官方安装文档：[PyTorch](https://pytorch.org/get-started/locally/)、[Triton](https://triton-lang.org/main/getting-started/installation.html)。

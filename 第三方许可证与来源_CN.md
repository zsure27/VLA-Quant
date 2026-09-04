# 第三方代码与许可证说明

本目录不是对第三方项目重新授权。使用前应分别遵守对应仓库和模型/数据集的许可证：

- QVLA：`AutoLab-SAI-SJTU/QVLA`，固定提交见 `configs/versions.json`。
- OpenVLA-OFT：随 QVLA checkout 获取；`overlays/` 是历史实验使用的覆盖文件。
- LIBERO：`Lifelong-Robot-Learning/LIBERO`。
- LLM-AWQ：`mit-han-lab/llm-awq`。
- SmoothQuant：`mit-han-lab/smoothquant`。
- OpenVLA-OFT checkpoint：Hugging Face `moojink/openvla-7b-oft-finetuned-libero-spatial`。

本仓库没有复制 AWQ/SmoothQuant 的完整第三方源码，而由恢复脚本从固定提交获取。`overlays/` 与 `legacy/` 保留上游/历史代码原有注释和署名语境，不能因为本目录文档为中文就视为原创代码。

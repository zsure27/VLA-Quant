# 大模型与实验数据清单

以下文件曾位于 AutoDL，但当前服务器无法开机，本地也没有副本，因此 GitHub 备份不包含其字节内容：

| 类别 | 历史路径 | 状态 | 恢复方式 |
|---|---|---|---|
| OpenVLA-OFT Spatial 模型 | `models/openvla-7b-oft-finetuned-libero-spatial` | 缺失 | 从 Hugging Face 重新下载，并记录 revision |
| LIBERO 数据 | `data/` 或 TensorFlow Datasets 根目录 | 缺失 | 按 LIBERO/QVLA 官方说明重建 |
| 平衡校准样本 | `calib/action-space-balanced/libero-512/sample-*.npz` | 缺失 | 用 `qvla/extract_balanced_calibration.py` 从训练 split 重建 |
| AWQ/SQ W4 profile | `qvla/spatial/official-w4-vl/{awq,smoothquant}/*.w4.pt` | 缺失 | 用修复后的校准器重新生成 |
| 原始评测日志 | `logs/official-quant-validation/*.log` | 缺失 | 只能重新运行；已保存汇总 CSV |

## 重新生成 512 个样本

准备好四套 `*_no_noops/1.0.0` TensorFlow Datasets 后运行：

```bash
cd /root/autodl-tmp/qvla-repro/src/QVLA/openvla-oft
python qvla/extract_balanced_calibration.py \
  --data-root /实际/TFDS/根目录 \
  --output-dir /root/autodl-tmp/qvla-repro/calib/action-space-balanced/libero-512 \
  --per-suite 128 --seed 7
```

这会得到四个 suite 各 128 个 episode 的单帧样本，共 512 个 `.npz`。它是本项目脚本定义的校准集，不要把“512 帧”写成论文原文的“512 条轨迹”。

## 恢复旧服务器后如何补备份

先执行 `scripts/inventory_server_data.sh` 生成文件名、大小与 SHA256 清单。大权重和数据集优先放可信对象存储；小型 profile/日志可放 GitHub Release，或在确认 Git LFS 配额后使用 LFS。仓库只提交清单、下载地址和校验值，不提交令牌、Cookie、`.env` 或 Hugging Face 缓存。

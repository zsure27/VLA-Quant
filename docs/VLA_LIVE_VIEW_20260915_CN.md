# 014 实时查看方法

VS Code 安装 Microsoft Remote - SSH 扩展后，添加以下 SSH host（本地的专用私钥已验证可连接 014）。这只是配置示例，不含私钥或密码：

```sshconfig
Host autodl-vla-014
    HostName connect.nmb1.seetacloud.com
    User root
    Port 19111
    IdentityFile C:/Users/zsure/.ssh/id_ed25519_vla_014
    IdentitiesOnly yes
    StrictHostKeyChecking yes
```

连接 `autodl-vla-014` 后打开 `/root/VLA-Quant` 看代码，打开 `/root/autodl-tmp/qvla-repro` 看结果。AutoDL JupyterLab 的 Terminal 中也能执行完全相同的远端命令；两个终端都是同一台机器，不要重复启动相同实验。

```bash
watch -n 2 bash /root/autodl-tmp/qvla-repro/scripts/vla_watch.sh
```

```bash
tail -f /root/autodl-tmp/qvla-repro/eval/diagnostic-queue.log
```

最新诊断根目录记录在 `/root/autodl-tmp/qvla-repro/eval/LATEST_DIAGNOSTIC.txt`。其下 `bf16/w4/language/vision` 每组含 `console.log`、`EVAL-*.txt`、`policy-queries.jsonl`、复现命令、退出码和哈希。网页监控页可以看 GPU 利用率/显存，但不能代替逐任务成功率。JupyterLab 文件列表可直接打开原始日志、JSON 或 PNG；文件变化后刷新。

本轮结束后这些路径继续保存在持久盘；关机期间 SSH 和实时刷新不可用，仍可查看本地或 GitHub 上的已同步结果。`Ctrl+C` 退出本文件给出的 watch/tail 只停止观察，不停止后台诊断队列；本轮队列使用 nohup 启动，每个子实验有超时限制，遇到执行或回合错误会停止继续派发。

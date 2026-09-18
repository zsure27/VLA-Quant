# AutoDL 014 快速 SSH 会话与备份（2026-09-15）

014 实例为 `3fbf46b812-2fdd6883`，SSH `root@connect.nmb1.seetacloud.com:19111`。本地专用 Ed25519 公钥已加入其 `authorized_keys`，`BatchMode` 免密连接实际通过；4090 在本轮开始时空闲。私钥仅在 Windows 用户 SSH 目录，未写入仓库。031 为另一个已克隆实例（SSH 端口 16917）；切换时明确指定 `-SshPort 16917` 并先核对实例 ID 和 GPU。脚本不会自动开机或关机。

在仓库目录中直接运行：

```powershell
./scripts/vla_fast_local.ps1 -Action check
./scripts/vla_fast_local.ps1 -Action smoke
./scripts/vla_fast_local.ps1 -Action baseline
./scripts/vla_fast_local.ps1 -Action w4
./scripts/vla_fast_local.ps1 -Action backup
```

本地包装器通过受信任主机密钥和专用公钥上传脚本，禁止密码交互。远端 `check` 核对 `zsure27/VLA-Quant` URL、基准提交祖先、校准 profile、robosuite 1.4.0、MuJoCo 3.1.1；`smoke` 跑 AWQ W2A16 当前候选，`baseline` 跑原始 BF16，`w4` 跑官方来源 AWQ W4A16。三组均为 LIBERO Spatial 10 任务各一回合、种子 0、配对协议，命令、原始日志、退出码和校验和写入服务器持久盘。当前逐任务结果与限制见 `docs/AWQ_PAIRED_SMOKE_20260915_CN.md`。

`backup` 对当前 Git 提交生成 bundle 和补丁，复制持久盘的原始 eval 文件与仓库 rollout 视频，逐文件核对，记录运行环境与未上传普通 Git 的大文件说明。非交互推送前必须以 `gh api user` 核对认证为 **zsure27** 并核对仓库 `permissions.push`；认证失败则保留全部备份并报告待推送。本地包装器再次复制 bundle、日志和视频，逐项核对 SHA-256。GitHub 远端提交仍需在推送后核对；remote URL 本身不证明账号身份。

本轮早期的模拟器初始化失败日志 `/root/autodl-tmp/qvla-repro/eval/awq-current-smoke10-20260915-103102/console.log` 无回合成功率。后续环境已锁定；014 上开始时的代码为 `852d350`，该提交此前尚未推送 GitHub。代码与当前原始结果已另存于持久盘 `/root/autodl-tmp/qvla-repro/backups/20260915-852d350` 和本地工作区备份；在 GitHub 核对之前不得把它称为完整云备份。

AutoDL 控制台的开关机和网页退出仍需可核验的浏览器工具或官方接口；SSH 内的 OS `shutdown` 不能证明计费实例关闭。额度任一窗口剩余不超过 10% 时停止新实验，至少留 3% 处理备份、关机和网页退出。额度接口若不可用，须提前收尾，不能假设剩余额度充足。

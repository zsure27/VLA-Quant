# 从私有 GitHub 归档恢复文件

完整原始归档保存于私有仓库 [`zsure27/VLA-Quant-Archives`](https://github.com/zsure27/VLA-Quant-Archives) 的 Release 附件。恢复需要该仓库的读取权限；公开代码仓库里的索引不代表原始数据公开。

## 1. 获取恢复目录并选择归档

以下命令在公开代码库 `VLA-Quant` 根目录运行，需已安装 Git 和 Python 3。先用有私有库读取权限的账号配置 Git Credential Manager，再获取恢复目录。浅克隆只获取当前目录、清单和脚本，不下载 Release 大附件；已有检出时更新它：

```powershell
$indexDir = '.local/private-archive-index'
if (Test-Path -LiteralPath "$indexDir/.git") {
    git -C $indexDir pull --ff-only
} else {
    git clone --depth 1 https://github.com/zsure27/VLA-Quant-Archives.git $indexDir
}
$manifest = Join-Path $indexDir 'catalog/20261017/remote-verification.json'
git -C $indexDir log -1 --format=%H
python scripts/restore_github_archive.py --manifest $manifest --list
```

记录上面显示的目录提交 SHA，便于复核。`--list` 只读本地 JSON，不联网或获取凭据；输出包含归档的完整 SHA、原文件名、原路径、大小和块数。恢复清单仅列该已发布检查点中通过完整核验的对象。原始路径用于查找材料，恢复时自行选择新位置。

## 2. 恢复一个文件

从清单中复制所需归档的完整 SHA256，选择尚不存在的输出路径。父目录须已创建，磁盘须能容纳完整文件。例如先建 `D:/VLA-Restore`，再按提示输入所选 SHA 和输出文件路径：

```powershell
New-Item -ItemType Directory -Path 'D:/VLA-Restore' -Force | Out-Null
$archiveSha256 = Read-Host '输入清单中选定归档的完整64位SHA256'
$destination = Read-Host '输入新的完整输出路径，例如 D:/VLA-Restore/complete.tar.gz'
python scripts/restore_github_archive.py --manifest $manifest --sha256 $archiveSha256 --destination $destination
```

脚本使用 `git credential fill` 取得凭据，仅向固定 GitHub API 发送令牌；跳转到 GitHub 签名下载地址时不转发令牌。每块按顺序流式写入 `complete.tar.gz.partial`，校验分块字节数与 SHA256，再校验整包。全部通过后才原子发布为目标文件，不覆盖已有文件，不自动解压。

下载失败时保留 `.partial` 便于检查；该文件不能当作有效备份。脚本不会覆盖或自动续写它，重试应选择新的目标文件名。只需恢复正在使用的材料，无须把全部历史归档重新下载到本机。

## 3. 查找旧路径或恢复展开文件

| 所需材料 | 私有目录中的入口 |
|---|---|
| 最初审计的完整 tar / bundle 与重复别名 | `catalog/20261017/original-archives.json` |
| 后续发现的旧备份 | `catalog/20261017/extra-backup-manifest-20261018.json` |
| 已清理的展开文件 | `catalog/20261017/coverage/` 中对应批次的成员链、大小和 SHA |
| B5 历史下载区间 | `catalog/20261017/range-staging-metadata-20261018.json` |

展开文件先按映射恢复其完整外层归档，再按成员链逐层提取到自行选择的恢复目录，最后核对该文件的大小和 SHA。`!` 是成员链的说明分隔符。脚本只还原归档字节，不自动解压或把历史绝对路径作为提取目标。

B5 历史区间仅保存当时下载到的后半部分，原完整包前缀缺失。恢复这些区间仍是部分原始材料，不能据此宣称完整 B5 包已恢复。迁移总进度和已核验目录提交见[索引首页](README_CN.md)。

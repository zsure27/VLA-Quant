# 从私有 GitHub 归档恢复文件

完整原始归档保存于私有仓库 [`zsure27/VLA-Quant-Archives`](https://github.com/zsure27/VLA-Quant-Archives) 的 Release 附件。恢复需要该仓库的读取权限；公开代码仓库里的索引不代表原始数据公开。

## 1. 选择归档

从私有归档库获取小型恢复索引。下面的克隆只获取目录文件，不下载 Release 大附件；已有检出时先更新它：

```powershell
git clone https://github.com/zsure27/VLA-Quant-Archives.git .local/private-archive-index
python scripts/restore_github_archive.py --manifest .local/private-archive-index/catalog/20261017/remote-verification.json --list
```

列清单只读本地 JSON 回执，不联网，也不获取凭据。原始文件路径与归档 SHA 的对应关系见私有库 `catalog/20261017/original-archives.json`；额外旧备份和 B5 不完整分段分别有独立映射。迁移期间，回执仅列已完成全部分块校验的对象。

## 2. 恢复一个文件

先在 Git Credential Manager 中配置有权限的 GitHub 账号。指定完整归档的 SHA256 和一个尚不存在的目标文件路径；父目录须已创建，磁盘须能容纳完整文件：

```powershell
python scripts/restore_github_archive.py --manifest .local/private-archive-index/catalog/20261017/remote-verification.json --sha256 <完整归档的64位SHA256> --destination D:/VLA-Restore/complete.tar.gz
```

脚本使用 `git credential fill` 取得凭据，仅向固定 GitHub API 发送令牌；跳转到 GitHub 签名下载地址时不转发令牌。每块按顺序流式写入 `complete.tar.gz.partial`，校验分块字节数与 SHA256，再校验整包。全部通过后才原子发布为目标文件，不覆盖已有文件，不自动解压。

下载失败时保留 `.partial` 便于检查；该文件不能当作有效备份。脚本不会覆盖或自动续写它，重试应选择新的目标文件名。只需恢复正在使用的材料，无须把全部历史归档重新下载到本机。

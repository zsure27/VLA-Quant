# 完整实验归档约定

完整本地归档统一放在以下位置，并由 `.gitignore` 排除：

```text
backups/experiments/<module>/YYYYMMDD-<host-or-scope>-<experiment>/<archive-id>/
```

这里保存 Git bundle、代码 patch、环境锁、压缩后的完整结果、视频/大文件清单、SHA256 和恢复说明。适合 Git 的小型证据仍抽取到同名的 `results/experiments/<module>/<session>/`，分析与图表进入 `reports/experiments/<module>/<session>/`。

`close_vla_session.ps1` 要求显式传入 `-ExperimentModule`，并核对报告与结果会话名一致。归档目录不得包含凭据；没有第二份原件和哈希核验时，不称为完整异地备份。

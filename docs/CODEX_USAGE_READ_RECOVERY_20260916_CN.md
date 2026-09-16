# 额度读数恢复与备用路径

用户反馈：107本轮关机时仍剩15%，这是用户事后报告，不冒充当时API读数。

2026-09-16日志显示`account/rateLimits/read`此前多次27～40秒后失败，后来806ms成功；主工具重新返回五小时剩14%、周剩78%。底层网络/服务间歇故障尚未确证，未修改登录、代理或应用二进制。官方字段和接口：[App Server文档](https://learn.chatgpt.com/docs/app-server)。

新增`scripts/codex_usage_read.py`：读取当前任务的`token_count.rate_limits`响应元数据，严格核对`session_meta.id`，支持同任务分段文件；不输出消息/凭据，不启动模型或AutoDL。保留真实时间戳，任一窗口缺失或记录超过300秒就返回UNKNOWN；剩余<10%或下一测试预算不能保留至少3%则要求备份关机。缓存仅保存额度及身份哈希，`.local/`不进Git。

实测备用读取1.03秒：五小时剩5%、周剩77%，数据年龄2.4秒；五项单元检查通过。本轮排查本身消耗了额度，不能把该读数当作此前关机时剩余。独立CLI读取不能核验账号身份，未认证为可用备用；默认不走它。

未来每两短/一长先运行：

```powershell
..\vla-audit-test-env\Scripts\python.exe scripts/codex_usage_read.py --thread-id 当前任务UUID --session-root C:/Users/zsure/.codex/sessions
```

有效新鲜读数按原10%规则执行；UNKNOWN再查询主工具一次。主工具失败时重读本任务响应元数据，仍无有效读数则保守收尾，禁止无限重复慢查询或把缓存伪装实时。当前任务UUID不能在新对话中照抄。

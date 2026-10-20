# router_dev 数据契约结果

**SUPPORTED（数据契约）**：固定 v3 轨迹分割 SHA256 `3560425653ef4b297a48a3854aae20348898129b6673db1474943e95897606a2`，按预注册五个时间位置抽取 `role=router_dev` 的 59 条轨迹、295 帧。输出 manifest SHA256 `0d42a53a07412f2839e0c30aee0ae606e914a3e3707748f1a577ba8e2370d364`；每帧含轨迹 ID、原始步索引、源路径、指令和 SHA。提取程序逐轨迹验证稳定 ID，并报告 `holdout_touched=false`。

允许的结论是：可重复的开发集合已准备。不能将其称为 final，不能据此声称任何模型泛化或闭环收益。下一问题是同一组295帧上 exact12L 与 A/B/C/D 的轨迹级配对分布；该评测已独立预注册并由顺序 runner 执行。

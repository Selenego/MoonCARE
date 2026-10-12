# S14 审计脚本 v1.1 修正说明

本版只修正评测脚本本身，不修改 MoonCARE 业务代码。

修正：
1. `.env.example` / `.env.sample` / `.env.template` 不再被误判为真实 secret 文件；
2. 源码扫描排除 `MoonCAREpack`、`.git`、`dist`、`build` 等重复/生成目录，避免旧副本造成误报。

原有的“整账号删除 API”和“会话删除 API”缺口不会被隐藏，仍会如实报告。

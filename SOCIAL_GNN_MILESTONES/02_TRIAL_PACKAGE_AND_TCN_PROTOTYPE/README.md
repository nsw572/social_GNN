# 02 — Full-trial 数据契约与 TCN 原型

时间：2026-09-07；Git 基线：`f0e3acd`。

## 这一步解决了什么

- 引入 `SocialTrialPackage`，严格验证 node、edge、身份和 patch clock 的一致性。
- 建立 PyTorch `Dataset`/`DataLoader`：一个 sample 就是一整场 trial；只在 batch 内 pad 到最长 T；生成 `time_mask`；绝不跨 trial 拼接时间。
- 为之后动态 SSL masking 保留原始 target，不修改存储文件。
- 当时还实现了 causal TCN，作为 `GNN -> TCN` 的首个时序原型。

## 当前地位

`social_gnn/data.py` 和 full-trial batching 契约保留并成为正式基础。TCN 后来被明确移除：encoder 只负责 patch-local `g_t`，时间结构交给冻结表示之后的 HMM/HSMM。历史 TCN 文件也已从当前主线删除。

## 关键代码

- 当前：`social_gnn/data.py`
- 历史快照：`git show f0e3acd:social_gnn/temporal.py`
- 数据层测试：`tests/test_data.py`

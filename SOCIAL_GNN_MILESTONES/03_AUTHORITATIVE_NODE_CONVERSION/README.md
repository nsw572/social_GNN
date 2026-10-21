# 03 — 权威 clock 与 canonical node conversion

时间：2026-09-19；Git 基线：`dc0f4d7`。

## 这一步解决了什么

- 建立 H ZIP → canonical node NPZ converter。
- 读取并传播权威 `patch_start_s` / `patch_end_s`，而不是由 edge 或模型私自重建时间轴。
- 对 node、edge 使用同一物理 patch interval，并严格检查 trial ID、identity order、shape、finite values 和 clock equality。
- 允许边界缺失 patch 保留在权威 grid 中、通过 mask 标无效，而不是偷偷缩短 trial。

## 关键代码

- `social_gnn/h_zip_node_converter.py`
- `social_gnn/authoritative_clock.py`
- 后续 6D/9D latent 正式转换器：`social_gnn/latent_node_converter.py`
- `tests/test_h_zip_converter.py`
- `tests/test_latent_node_converter.py`

这一步奠定了后来“0.16 s 形式 patch 可以有重叠感受野，但所有模态必须指向同一权威 interval”的数据原则。

# # video_diffusion/prompt_attention/flow_traj_attn.py
# -*- coding: utf-8 -*-
# import math
# import torch
# from einops import rearrange

# # ---------------------------------------------
# # Helpers
# # ---------------------------------------------
# def _sreg_full_to_per_frame(sreg_full, F_clip, N, B, device):
#     """
#     sreg_full: [Bhz, FN, FN]  (舊格式；Bhz 可能是 1、B，或 B*H 等)
#     轉成     : [B, F, N, N]   (新格式；只保留「同幀」的 N×N 區塊)

#     規則：
#       - 若 Bhz == 1 → expand 到 B
#       - 若 Bhz == B → 直接使用
#       - 若 Bhz 不是 1 或 B（例如 B*H）→ 先 reshape 成 [B, -1, FN, FN]，沿頭數維度做 amax（或 any）
#     """
#     assert sreg_full.dim() == 3 and sreg_full.size(1) == F_clip * N and sreg_full.size(2) == F_clip * N, \
#         f"Expect [Bhz, FN, FN] with FN={F_clip*N}, got {tuple(sreg_full.shape)}"

#     Bhz, FN1, FN2 = sreg_full.shape
#     assert FN1 == FN2 == F_clip * N

#     # 把不同頭/通道聚合到 batch：若 Bhz 不是 1 或 B，視為 [B, H*, FN, FN]，沿 H* 聚合為任一>0 即算 1（或取最大）
#     if Bhz == 1:
#         sreg_b = sreg_full.expand(B, FN1, FN2)
#     elif Bhz == B:
#         sreg_b = sreg_full
#     else:
#         if Bhz % B != 0:
#             raise RuntimeError(f"Cannot map sreg batch {Bhz} to B={B}.")
#         Hstar = Bhz // B
#         sreg_b = sreg_full.view(B, Hstar, FN1, FN2).amax(dim=1)  # 等價於「任一 head 為真即為真」

#     sreg_b = (sreg_b > 0)

#     # 擷取「同幀」方塊，產生 [B, F, N, N]
#     out = torch.empty((B, F_clip, N, N), dtype=torch.bool, device=device)
#     for f in range(F_clip):
#         r = slice(f * N, (f + 1) * N)
#         out[:, f] = sreg_b[:, r, r]
#     return out

# def reshape_heads_to_batch_dim3(x, heads):
#     """
#     x: [(B*F), N, L(或1), D] -> [(B*F), H, N, L(或1), d_h]
#     """
#     Bf, N, L, D = x.shape
#     assert D % heads == 0, f"D={D} must be divisible by heads={heads}"
#     d_h = D // heads
#     x = x.view(Bf, N, L, heads, d_h)
#     x = x.permute(0, 3, 1, 2, 4).contiguous()
#     return x

# def normalize_traj_and_mask(traj, mask, F_clip, N):
#     """
#     將輸入的 traj / mask 正規化到：
#       traj: [F, N, L, 3]  (t,x,y)
#       mask: [F, N, L]     (bool/int)
#     支援下列形狀的自動轉換：
#       - traj: [F,N,L,3] 或 [N,F,L,3] 或 [(F*N),L,3]
#       - mask: [F,N,L]   或 [N,F,L]   或 [(F*N),L]
#     """
#     # traj -> [F, N, L, 3]
#     if traj.dim() == 4:
#         if traj.shape[0] == F_clip and traj.shape[1] == N:
#             pass
#         elif traj.shape[0] == N and traj.shape[1] == F_clip:
#             traj = traj.permute(1, 0, 2, 3).contiguous()
#         else:
#             raise ValueError(f"Unexpected traj 4D shape: {traj.shape}, expected [F,N,L,3] or [N,F,L,3]")
#     elif traj.dim() == 3:
#         if traj.shape[0] == F_clip * N:
#             traj = rearrange(traj, '(f n) l d -> f n l d', f=F_clip, n=N)
#         else:
#             raise ValueError(f"Unexpected traj 3D shape: {traj.shape}, expected [(F*N),L,3]")
#     else:
#         raise ValueError(f"Unexpected traj dim: {traj.dim()} with shape {traj.shape}")

#     # mask -> [F, N, L]
#     if mask.dim() == 3:
#         if mask.shape[0] == F_clip and mask.shape[1] == N:
#             pass
#         elif mask.shape[0] == N and mask.shape[1] == F_clip:
#             mask = mask.permute(1, 0, 2).contiguous()
#         else:
#             raise ValueError(f"Unexpected mask 3D shape: {mask.shape}, expected [F,N,L] or [N,F,L]")
#     elif mask.dim() == 2:
#         if mask.shape[0] == F_clip * N:
#             mask = rearrange(mask, '(f n) l -> f n l', f=F_clip, n=N)
#         else:
#             raise ValueError(f"Unexpected mask 2D shape: {mask.shape}, expected [(F*N),L]")
#     else:
#         raise ValueError(f"Unexpected mask dim: {mask.dim()} with shape {mask.shape}")

#     return traj, mask

# # ---------------------------------------------
# # Main: Flow + Same-Semantics Full-History Attention (No Top-K)
# # ---------------------------------------------
# @torch.no_grad()  # 你如果要訓練，請移除此 decorator
# def flow_semantic_traj_attention(
#     query_old, key_old, value_old,
#     encoder_hidden_states, group_norm,
#     traj, mask, time_causal,
#     _key, _value,                 # [B, F, H, W, D]
#     h, w, clip_length, heads,     # spatial & heads
#     controller, sem_chunk=128,    # chunk over same-semantics C 維度
#     use_sem_aug=True,
#     flow_only=False,
#     old_qk=1
# ):
#     """
#     精確等價於「每個像素 Query 看：
#       (1) 自己的過去軌跡 (self-traj, 長度 L)
#       (2) 同幀同語意像素及其各自的過去軌跡（完整、不 top-K）
#     」，並作時間因果 (<= anchor frame)。

#     重要差異（效能）：
#       - sreg 使用 per-frame: controller.sreg_maps[h*w] 形狀需為 [B,F,N,N] 或 [1,F,N,N]
#       - 移除 repeat_interleave(L) 與 CL 展開；改以 C×L 二維同時加總的 streaming softmax
#       - 無近似，結果與一次性對 (self + same-semantics full history) 做 softmax 相同
#     """
#     device = encoder_hidden_states.device
#     base_dtype = key_old.dtype
#     acc_dtype  = torch.float32 if base_dtype == torch.float16 else base_dtype

#     BxF, seq_len, Cdim = encoder_hidden_states.shape
#     F_clip = clip_length
#     H, W = h, w
#     N = H * W
#     B = BxF // F_clip
#     assert (B * F_clip) == BxF, "Batch/clip length mismatch"

#     # --- choose q/k/v source ---
#     if old_qk == 1:
#         query = query_old
#         key   = key_old
#     else:
#         query = encoder_hidden_states
#         key   = encoder_hidden_states
#     value = encoder_hidden_states if not flow_only else value_old

#     # group norm（若外部沒做，這裡保持一致）
#     if group_norm is not None:
#         encoder_hidden_states = group_norm(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)

#     # ---- 正規化 traj, mask：得到 [F,N,L,*] ----
#     traj, mask = normalize_traj_and_mask(traj, mask, F_clip, N)  # [F,N,L,3], [F,N,L]
#     t_inds = traj[..., 0].long()  # [F,N,L]
#     x_inds = traj[..., 1].long()
#     y_inds = traj[..., 2].long()

#     # time-causal
#     anchor = t_inds[:, :, 0].unsqueeze(-1).expand_as(t_inds)  # [F,N,L]
#     if time_causal:
#         traj_mask = t_inds <= anchor
#     else:
#         traj_mask = torch.ones_like(t_inds, dtype=torch.bool)
#     t_inds = torch.where(traj_mask, t_inds, torch.zeros_like(t_inds))
#     x_inds = torch.where(traj_mask, x_inds, torch.zeros_like(x_inds))
#     y_inds = torch.where(traj_mask, y_inds, torch.zeros_like(y_inds))

#     # 將 (t,x,y) 改成 flatten index，便於處理「同幀語意 vs 自身軌跡去重」
#     flat_traj = (t_inds * N + (x_inds * W + y_inds))  # [F,N,L]

#     # ---- 準備 Q / 沿軌跡的 K/V ----
#     # Q: [(B*F), N, 1, D] -> [(B*F), H, N, 1, d_h]
#     query_tempo = query.unsqueeze(-2)  # [(B*F), N, 1, D]
#     q_h = reshape_heads_to_batch_dim3(query_tempo, heads=heads).to(acc_dtype)

#     # K/V 沿軌跡 gather，先得到 [B,F,N,L,D] 再展成 [(B*F),N,L,D]
#     # _key/_value: [B,F,H,W,D]
#     with torch.autocast(device_type=str(device).split(':')[0], enabled=False):
#         key_tempo   = _key[:, t_inds, x_inds, y_inds]    # [B, F, N, L, D]
#         value_tempo = _value[:, t_inds, x_inds, y_inds]  # [B, F, N, L, D]
#     key_tempo   = rearrange(key_tempo,   'b f n l d -> (b f) n l d')
#     value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d')

#     kt_h = reshape_heads_to_batch_dim3(key_tempo,   heads=heads).to(acc_dtype)  # [(B*F),H,N,L,d_h]
#     vt_h = reshape_heads_to_batch_dim3(value_tempo, heads=heads).to(acc_dtype)  # [(B*F),H,N,L,d_h]

#     # ---- sreg: per-frame [B,F,N,N] ----
#     # ---- sreg: 接受兩種格式 ----
#     if use_sem_aug and controller.sreg_maps[h * w] is not None:
#         sreg_any = controller.sreg_maps[h * w]

#         if sreg_any.dim() == 4 and sreg_any.size(0) in (1, B) and \
#            sreg_any.size(1) == F_clip and sreg_any.size(2) == N and sreg_any.size(3) == N:
#             # 已是 per-frame: [B,F,N,N] 或 [1,F,N,N]
#             if sreg_any.size(0) == 1 and B > 1:
#                 sreg_pf = sreg_any.expand(B, F_clip, N, N).contiguous()
#             else:
#                 sreg_pf = sreg_any
#             sreg_pf = (sreg_pf > 0)

#         elif sreg_any.dim() == 3 and sreg_any.size(1) == F_clip * N and sreg_any.size(2) == F_clip * N:
#             # 舊格式: [Bhz, FN, FN]  → 轉成 [B, F, N, N]
#             sreg_pf = _sreg_full_to_per_frame(sreg_any.to(encoder_hidden_states.device),
#                                               F_clip, N, B, encoder_hidden_states.device)
#         else:
#             raise RuntimeError(
#                 f"Unsupported sreg shape for F={F_clip}, N={N}: got {tuple(sreg_any.shape)}"
#             )
#     else:
#         sreg_pf = None


#     # ---- Streaming Softmax 容器 ----
#     d_h = q_h.size(-1)
#     scale = torch.tensor(1.0 / math.sqrt(d_h), dtype=acc_dtype, device=device)
#     out_all = torch.empty((B * F_clip, heads, N, d_h), device=device, dtype=base_dtype)

#     # ---- 逐幀處理（外層 loop 只剩 F 級；內部消除了巨量展開）----
#     for fcur in range(F_clip):
#         bf_idx = (torch.arange(B, device=device) * F_clip + fcur)   # [B]
#         q_h_f  = q_h[bf_idx]    # [B,H,N,1,d_h]
#         kt_f   = kt_h[bf_idx]   # [B,H,N,L,d_h]（每個像素自己的 traj）
#         vt_f   = vt_h[bf_idx]   # [B,H,N,L,d_h]

#         # (1) Self-traj 注意力：對 [L] 做精確 softmax（初始化 streaming 狀態）
#         # logits_traj: [B,H,N,1,L]
#         logits_traj = torch.matmul(q_h_f * scale, kt_f.transpose(-2, -1))  # K^T over L: [...,1,d] x [...,d,L]
#         # 依照 mask 限制：keep_mask: [F,N,L] -> 當幀 fcur 的 [B,N,L]
#         keep_mask = (mask[fcur].to(torch.bool) & traj_mask[fcur])  # [N,L]（無 batch）
#         keep_mask_b = keep_mask.unsqueeze(0).expand(B, N, keep_mask.size(-1))  # [B,N,L]
#         # 轉成 [B,H,N,1,L] 做遮罩
#         attn_mask = keep_mask_b[:, None, :, None, :]  # [B,1,N,1,L]
#         # neg_inf = torch.finfo(acc_dtype).min
#         # logits_traj = torch.where(attn_mask, logits_traj, torch.full_like(logits_traj, neg_inf))

#         neg_val = torch.finfo(logits_traj.dtype).min  # 用 logits 的 dtype 取範圍
#         logits_traj = logits_traj.masked_fill(~attn_mask, neg_val)

#         # softmax-初始化
#         m  = torch.max(logits_traj, dim=-1, keepdim=True).values  # [B,H,N,1,1]
#         exp_logits = torch.exp(logits_traj - m)                   # [B,H,N,1,L]
#         Z   = exp_logits.sum(dim=-1, keepdim=True)                # [B,H,N,1,1]
#         Out = torch.matmul(exp_logits, vt_f)                      # [B,H,N,1,d_h]

#         # (2) 同幀同語意 + 過去軌跡（完整 L、無 top-K）

#         # (2) 同幀同語意 + 過去軌跡（完整 L、無 top-K）
#         if sreg_pf is not None:
#             # 取同幀 [B,N,N]；去掉「自身軌跡在當幀的位置」避免重複
#             same_sem_mask = sreg_pf[:, fcur]  # [B,N,N] (bool)
#             traj_cols_this = (flat_traj[fcur] % N)  # [N,L]（無 batch）

#             # 建立去重遮罩 M: [B,N,N]
#             M = torch.zeros((B, N, N), dtype=torch.bool, device=device)
#             idx_scatter = traj_cols_this.unsqueeze(0).expand(B, -1, -1)  # [B,N,L]
#             M.scatter_(dim=2, index=idx_scatter, src=torch.ones_like(idx_scatter, dtype=torch.bool, device=device))
#             same_sem_mask = same_sem_mask & (~M)  # [B,N,N]

#             # 小工具：把張量整理成 [B,H,N,1,1]（若是 [B,N,H,1,1] 則交換 dim1/2）
#             def _as_BHN11(x, name):
#                 if x.dim() != 5 or x.size(0) != B or x.size(3) != 1 or x.size(4) != 1:
#                     raise RuntimeError(f"{name} shape {tuple(x.shape)} invalid; expect 5D [B,*,*,1,1]")
#                 if x.size(1) == heads and x.size(2) == N:
#                     return x
#                 if x.size(1) == N and x.size(2) == heads:
#                     return x.transpose(1, 2)
#                 raise RuntimeError(f"{name} bad shape {tuple(x.shape)}; expect [B,H,N,1,1] or [B,N,H,1,1]")

#             # 以 C 維（同語意像素）切塊，避免一次把 N×L 攤平成 CL
#             C = sem_chunk if sem_chunk > 0 else N
#             for s in range(0, N, C):
#                 e = min(s + C, N)
#                 # 取當前塊的 C 個來源像素：[B,N,C]
#                 mask_chunk = same_sem_mask[:, :, s:e]  # [B,N,C] (bool)

#                 # 取對應的 K/V：Kc,Vc: [B,H,C,L,d_h]
#                 Kc = kt_f[:, :, s:e, :, :]  # [B,H,C,L,d_h]
#                 Vc = vt_f[:, :, s:e, :, :]  # [B,H,C,L,d_h]

#                 # ---- 防呆檢查 ----
#                 assert q_h_f.dim() == 5 and q_h_f.size(0) == B and q_h_f.size(1) == heads \
#                        and q_h_f.size(2) == N and q_h_f.size(3) == 1, \
#                        f"q_h_f shape {tuple(q_h_f.shape)} != [B,H,N,1,d]"
#                 assert Kc.dim() == 5 and Kc.size(0) == B and Kc.size(1) == heads \
#                        and Kc.size(2) == (e - s), f"Kc shape {tuple(Kc.shape)} mismatch with chunk C={e-s}"
#                 assert Vc.shape == Kc.shape, f"Vc {tuple(Vc.shape)} != Kc {tuple(Kc.shape)}"
#                 assert mask_chunk.shape == (B, N, e - s), \
#                        f"mask_chunk {tuple(mask_chunk.shape)} != (B,N,C={(e-s)})"

#                 # logits_c（不展開成 CL！保留 C 與 L 兩維）
#                 # q_h_f : [B,H,N,1,d]；Kc: [B,H,C,L,d] → logits_c5: [B,H,N,1,C,L]
#                 logits_c5 = torch.einsum('bhnrd,bhcld->bhnrcl', q_h_f * scale, Kc)

#                 # 遮罩（用 logits 的 dtype；避免 half 溢位）
#                 mask_b = mask_chunk[:, None, :, None, :, None]  # [B,1,N,1,C,1]
#                 neg_val_c = torch.finfo(logits_c5.dtype).min
#                 logits_c5 = logits_c5.masked_fill(~mask_b, neg_val_c)

#                 # 串流 softmax 合併：沿 (C,L) 一次性合併，而非攤平成 CL
#                 # 先求此塊的最大值 -> 期望 [B,H,N,1,1]
#                 m_c = torch.amax(logits_c5, dim=(-2, -1), keepdim=True)   # over C,L
#                 # m_c = torch.amax(logits_c5, dim=(-2, -1), keepdim=True)  # 先計算
#                 m_c = m_c.squeeze(-2)  # 去掉多餘的那個 1

#                 # 將 m / m_c 整到 [B,H,N,1,1]，避免 H/N 對調
#                 m   = _as_BHN11(m,   "m(self-traj)")
#                 m_c = _as_BHN11(m_c, "m_c(chunk)")

#                 m_new = torch.maximum(m, m_c)                             # [B,H,N,1,1]
#                 alpha  = torch.exp(m - m_new)                             # [B,H,N,1,1]
#                 weights = torch.exp(logits_c5 - m_new)                    # [B,H,N,1,C,L]

#                 # 分母
#                 Z = _as_BHN11(Z, "Z(self-traj)")                          # ensure [B,H,N,1,1]
#                 Z = Z * alpha + torch.sum(weights, dim=(-2, -1), keepdim=True)  # [B,H,N,1,1]

#                 # 分子：對 Vc 沿 (C,L) 權重加總 → [B,H,N,1,d_h]
#                 out_inc = torch.einsum('bhnrcl,bhcld->bhnrd', weights, Vc)

#                 # 更新 Out / m
#                 Out = Out * alpha + out_inc                                # [B,H,N,1,d_h]
#                 m   = m_new

#         # if sreg_pf is not None:
#         #     # 取同幀 [B,N,N]；去掉「自身軌跡在當幀的位置」避免重複（等價於你原本的處理）
#         #     same_sem_mask = sreg_pf[:, fcur]  # [B,N,N] (bool)
#         #     traj_cols_this = (flat_traj[fcur] % N)  # [N,L]（無 batch）
#         #     # 建立去重遮罩 M: [B,N,N]
#         #     M = torch.zeros((B, N, N), dtype=torch.bool, device=device)
#         #     # 將每個像素 n 的 L 個當幀列（col）標 True
#         #     # scatter_: dim=2 (col), index:[B,N,L]
#         #     idx_scatter = traj_cols_this.unsqueeze(0).expand(B, -1, -1)  # [B,N,L]
#         #     M.scatter_(dim=2, index=idx_scatter, src=torch.ones_like(idx_scatter, dtype=torch.bool, device=device))
#         #     same_sem_mask = same_sem_mask & (~M)  # [B,N,N]

#         #     # 以 C 維（同語意像素）切塊，避免一次把 N×L 攤平成 CL
#         #     C = sem_chunk if sem_chunk > 0 else N
#         #     for s in range(0, N, C):
#         #         e = min(s + C, N)
#         #         # 取當前塊的 C 個來源像素：[B,N,C]
#         #         mask_chunk = same_sem_mask[:, :, s:e]  # [B,N,C] (bool)

#         #         # 取對應的 K/V：Kc,Vc: [B,H,C,L,d_h]
#         #         Kc = kt_f[:, :, s:e, :, :]  # [B,H,C,L,d_h]
#         #         Vc = vt_f[:, :, s:e, :, :]  # [B,H,C,L,d_h]

#         #         # logits_c（不展開成 CL！保留 C 與 L 兩維）
#         #         # q: [B,H,N,1,d] ; Kc: [B,H,C,L,d] -> logits_c5: [B,H,N,1,C,L]
#         #         # 使用愛因斯坦求和避免中間轉置
#         #         # b,h,n,one,d  x  b,h,c,l,d  ->  b,h,n,one,c,l
#         #         # logits_c5 = torch.einsum('bhqrd,bhcld->bhqrcl', q_h_f * scale, Kc)  # q 有 r=1
#         #         logits_c5 = torch.einsum('bhnrd,bhcld->bhnrcl', q_h_f * scale, Kc)

#         #         # 遮罩：將非法來源（非同語意或自軌跡重疊）位置設為 -inf
#         #         # mask_chunk: [B,N,C] -> [B,1,N,1,C,1]，沿 L 維自動 broadcast
#         #         mask_b = mask_chunk[:, None, :, None, :, None]  # [B,1,N,1,C,1]
#         #         # logits_c5 = torch.where(mask_b, logits_c5, torch.full_like(logits_c5, neg_inf))
#         #         neg_val_c = torch.finfo(logits_c5.dtype).min  # 用 logits_c5 的 dtype
#         #         logits_c5 = logits_c5.masked_fill(~mask_b, neg_val_c)

#         #         # 串流 softmax 合併：沿 (C,L) 一次性合併，而非攤平成 CL
#         #         # 先求此塊的最大值 -> [B,H,N,1,1]
#         #         m_c = torch.amax(logits_c5, dim=(-2, -1), keepdim=True)  # max over C,L
#         #         m_new = torch.maximum(m, m_c)  # [B,H,N,1,1]

#         #         # alpha = exp(m - m_new)，對 Out/Z 做縮放
#         #         alpha = torch.exp(m - m_new)  # [B,H,N,1,1]
#         #         # 新增權重
#         #         weights = torch.exp(logits_c5 - m_new)  # [B,H,N,1,C,L]
#         #         # 累加分母
#         #         Z = Z * alpha + torch.sum(weights, dim=(-2, -1), keepdim=True)  # [B,H,N,1,1]
#         #         # 累加分子（對 Vc 沿 C,L 權重和）
#         #         # weights: [B,H,N,1,C,L], Vc:[B,H,C,L,d] -> out_inc:[B,H,N,1,d]
#         #         out_inc = torch.einsum('bhqncl,bhcld->bhqnd', weights, Vc)
#         #         out_inc = out_inc.unsqueeze(-2)  # [B,H,N,1,d]
#         #         Out = Out * alpha + out_inc
#         #         m = m_new

#         # 正規化
#         out_f = (Out / (Z + 1e-12)).squeeze(-2)   # [B,H,N,d_h]
#         out_all[bf_idx] = out_f.to(base_dtype)

#     # 回到 [(B*F), (H*W), (H*d_h)] layout
#     hidden_states = rearrange(out_all, '(b f) h (H W) d -> b (f H W) (h d)', b=B, f=F_clip, H=H, W=W)
#     return hidden_states
# video_diffusion/prompt_attention/flow_traj_attn.py
# video_diffusion/prompt_attention/flow_traj_attn.py
import math
import torch
from einops import rearrange

def reshape_heads_to_batch_dim3(x, heads):
    # x: [(B*F), N, 1 or L, D] -> [(B*F), H, N, 1 or L, d_h]
    Bf, N, L, D = x.shape
    assert D % heads == 0, f"D={D} must be divisible by heads={heads}"
    d_h = D // heads
    x = x.view(Bf, N, L, heads, d_h)
    x = x.permute(0, 3, 1, 2, 4).contiguous()
    return x

def take_block(mask, s, e):
    return mask[..., s:e]

def normalize_traj_and_mask(traj, mask, F_clip, N):
    # traj -> [F, N, L, 3]
    if traj.dim() == 4:
        if traj.shape[0] == F_clip and traj.shape[1] == N:
            pass
        elif traj.shape[0] == N and traj.shape[1] == F_clip:
            traj = traj.permute(1, 0, 2, 3).contiguous()
        else:
            raise ValueError(f"Unexpected traj 4D shape: {traj.shape}, expected [F,N,L,3] or [N,F,L,3]")
    elif traj.dim() == 3:
        if traj.shape[0] == F_clip * N:
            traj = rearrange(traj, '(f n) l d -> f n l d', f=F_clip, n=N)
        else:
            raise ValueError(f"Unexpected traj 3D shape: {traj.shape}, expected [(F*N),L,3]")
    else:
        raise ValueError(f"Unexpected traj dim: {traj.dim()} with shape {traj.shape}")

    # mask -> [F, N, L]
    if mask.dim() == 3:
        if mask.shape[0] == F_clip and mask.shape[1] == N:
            pass
        elif mask.shape[0] == N and mask.shape[1] == F_clip:
            mask = mask.permute(1, 0, 2).contiguous()
        else:
            raise ValueError(f"Unexpected mask 3D shape: {mask.shape}, expected [F,N,L] or [N,F,L]")
    elif mask.dim() == 2:
        if mask.shape[0] == F_clip * N:
            mask = rearrange(mask, '(f n) l -> f n l', f=F_clip, n=N)
        else:
            raise ValueError(f"Unexpected mask 2D shape: {mask.shape}, expected [(F*N),L]")
    else:
        raise ValueError(f"Unexpected mask dim: {mask.dim()} with shape {mask.shape}")

    return traj, mask

'''
def flow_semantic_traj_attention(
    query_old, key_old, value_old,
    encoder_hidden_states, group_norm,
    traj, mask, time_causal,
    _key, _value,                 # [B, F, H, W, D]
    h, w, clip_length, heads,     # spatial & heads
    controller, sem_chunk=128,    # chunk for memory
    use_sem_aug=True,
    flow_only=False,
    old_qk=1
):'''
def flow_semantic_traj_attention(
    query_old, key_old, value_old,
    encoder_hidden_states, group_norm,
    traj, mask, time_causal,
    _key, _value,                 # [B, F, H, W, D]
    h, w, clip_length, heads,     # spatial & heads
    controller, sem_chunk=128,    # chunk for memory
    use_sem_aug=True,
    flow_only=False,
    old_qk=1,
    future_lookahead: int = 0,
    temporal_decay_tau: float = 1.0
):

    """
    每個 query 既看：自己過去的軌跡（self-traj），也看同幀同語意像素及其各自的過去軌跡（same-class past-traj）。
    """
    device = encoder_hidden_states.device
    base_dtype = key_old.dtype
    acc_dtype = torch.float32 if base_dtype == torch.float16 else base_dtype

    BxF, seq_len, C = encoder_hidden_states.shape
    F_clip = clip_length
    Bsmall = BxF // F_clip
    H, W = h, w
    N = H * W
    FN = F_clip * N

    # q/k/v 選擇
    if old_qk == 1:
        query = query_old
        key   = key_old
    else:
        query = encoder_hidden_states
        key   = encoder_hidden_states
    value = encoder_hidden_states if not flow_only else value_old

    # group norm（若外面有做可跳過，但這裡保持一致）
    if group_norm is not None:
        encoder_hidden_states = group_norm(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)

    # ---- 軌跡與 mask（均為 [F,N,L,*]）已於呼叫端正規化，這裡直接用 ----
    # t/x/y
    t_inds = traj[..., 0].long() #f,n l
    x_inds = traj[..., 1].long()
    y_inds = traj[..., 2].long()
    '''
    # time-causal
    anchor = t_inds[:, :, 0].unsqueeze(-1).expand_as(t_inds)  # [F,N,L]
    if time_causal:
        traj_mask = t_inds <= anchor
    else:
        traj_mask = torch.ones_like(t_inds, dtype=torch.bool)
    t_inds = torch.where(traj_mask, t_inds, torch.zeros_like(t_inds))
    x_inds = torch.where(traj_mask, x_inds, torch.zeros_like(x_inds))
    y_inds = torch.where(traj_mask, y_inds, torch.zeros_like(y_inds))
    '''
    
        # time-causal（允許短未來視窗）
    anchor = t_inds[:, :, 0].unsqueeze(-1).expand_as(t_inds)  # [F,N,L]
    if time_causal:
        upper = anchor + future_lookahead
        traj_mask = t_inds <= upper
    else:
        traj_mask = torch.ones_like(t_inds, dtype=torch.bool)

    # 越界時間索引改為 0（實際會被 mask 擋掉）
    t_inds = torch.where(traj_mask, t_inds, torch.zeros_like(t_inds))
    x_inds = torch.where(traj_mask, x_inds, torch.zeros_like(x_inds))
    y_inds = torch.where(traj_mask, y_inds, torch.zeros_like(y_inds))

    # 對「未來」點加時間衰減：dt = max(t - anchor, 0)
    dt = (t_inds - anchor).clamp(min=0)
    td_dtype = encoder_hidden_states.dtype
    time_decay = torch.exp(- dt.to(td_dtype) / max(temporal_decay_tau, 1e-6))  # [F,N,L]



    # flat 位置 id：同幀語義去重用
    flat_traj = (t_inds * N + (x_inds * W + y_inds)).unsqueeze(0).expand(Bsmall, F_clip, N, -1)  # [B,F,N,L]

    # Q
    query_tempo = query.unsqueeze(-2)  # [(B*F), N, 1, D]
    q_h = reshape_heads_to_batch_dim3(query_tempo, heads=heads).to(acc_dtype) #[(B*F), H, N, D]

    # 沿軌跡 gather K/V
    key_tempo   = _key[:, t_inds, x_inds, y_inds]    # [B, F, N, L, D]
    value_tempo = _value[:, t_inds, x_inds, y_inds]  # [B, F, N, L, D]
    key_tempo   = rearrange(key_tempo, 'b f n l d -> (b f) n l d')
    value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d')
    kt_h = reshape_heads_to_batch_dim3(key_tempo, heads=heads).to(acc_dtype)  # [(B*F), H, N, L, d_h]
    vt_h = reshape_heads_to_batch_dim3(value_tempo, heads=heads).to(acc_dtype) # [(B*F), H, N, L, d_h]

    # mask 只作用在 self-traj 部分
    keep_mask = mask.to(torch.bool) & traj_mask
    keep_mask_b = keep_mask.unsqueeze(0).expand(Bsmall, F_clip, N, keep_mask.size(-1))  # [B,F,N,L]
    mask_traj_bf = rearrange(keep_mask_b, 'b f n l -> (b f) n l')                       # [(B*F),N,L]
    attn_mask = mask_traj_bf[:, None].repeat(1, heads, 1, 1).unsqueeze(-2)              # [(B*F),H,N,1,L] (bool)

    # 流式 softmax 容器
    scale = torch.tensor(1.0 / math.sqrt(q_h.size(-1)), dtype=acc_dtype, device=device)
    out_all = torch.empty((Bsmall * F_clip, heads, N, q_h.size(-1)),
                          device=device, dtype=base_dtype)    #儲存每一幀計算完後的注意力

    # sreg
    if use_sem_aug and controller.sreg_maps[h * w] is not None:
        sreg_full = controller.sreg_maps[h * w]  # [B, FN, FN] or [1, FN, FN]
        assert sreg_full.dim() == 3 and sreg_full.size(1) == FN and sreg_full.size(2) == FN, \
            f"sreg shape mismatch: got {tuple(sreg_full.shape)}, expect (*,{FN},{FN})"
        if sreg_full.size(0) == 1:
            sreg_full = sreg_full.expand(Bsmall, FN, FN)
        elif sreg_full.size(0) != Bsmall:
            raise RuntimeError(f"Unexpected sreg batch dim: {sreg_full.size(0)} vs B={Bsmall}")
    else:
        sreg_full = None

    # 逐幀
    for fcur in range(F_clip):
        bf_idx = (torch.arange(Bsmall, device=device) * F_clip + fcur)  # [B]
        q_h_f  = q_h[bf_idx]               # [B,H,N,1,d_h]
        kt_h_f = kt_h[bf_idx]              # [B,H,N,L,d_h]
        vt_h_f = vt_h[bf_idx]              # [B,H,N,L,d_h]

        # (1) self-traj
        '''
        logits_traj = torch.matmul(q_h_f * scale, kt_h_f.transpose(-2, -1))  # [B,H,N,1,L]
        local_neg_inf = torch.finfo(acc_dtype).min
        bias_traj_f = torch.zeros_like(logits_traj, dtype=acc_dtype, device=device)
        bias_traj_f = torch.where(
            rearrange(attn_mask[bf_idx], 'b h n one l -> b h n one l'),
            bias_traj_f,
            torch.full_like(bias_traj_f, local_neg_inf)
        )
        logits_traj = logits_traj + bias_traj_f'''
        logits_traj = torch.matmul(q_h_f * scale, kt_h_f.transpose(-2, -1))  # [B,H,N,1,L]

        # + 時間衰減（未來項目）
        td_f = time_decay[fcur].unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(B, heads, N, 1, -1)
        logits_traj = logits_traj + torch.log(td_f + 1e-12)

        # 套 self-traj mask
        local_neg_inf = torch.finfo(acc_dtype).min
        bias_traj_f = torch.zeros_like(logits_traj, dtype=acc_dtype, device=device)
        bias_traj_f = torch.where(
            rearrange(attn_mask[bf_idx], 'b h n one l -> b h n one l'),
            bias_traj_f,
            torch.full_like(bias_traj_f, local_neg_inf)
        )
        logits_traj = logits_traj + bias_traj_f

        m  = torch.max(logits_traj, dim=-1, keepdim=False).values    # [B,H,N,1]
        exp_logits = torch.exp(logits_traj - m.unsqueeze(-1))        # [B,H,N,1,L]
        Z   = exp_logits.sum(dim=-1)                                 # [B,H,N,1]
        Out = torch.matmul(exp_logits, vt_h_f)                       # [B,H,N,1,d_h]

        # (2) 同幀同語意 + 其過去軌跡
        if sreg_full is not None:
            q_flat = fcur * N + torch.arange(N, device=device)       # [N] 該幀要問的query
            sreg_rows = torch.take_along_dim(
                sreg_full, q_flat.view(1, -1, 1).expand(Bsmall, N, FN), dim=1
            )                                                        # [B, N, FN]  #要問query對應的token關係(列)
            sreg_rows_frame = sreg_rows[:, :, fcur * N:(fcur + 1) * N]   # [B, N, N] 要問query對應的特定幀token
            same_sem_mask = (sreg_rows_frame > 0)                        # [B, N, N] , >0 為true

            # 把當前幀內，同語意像素遮罩中「落在自己軌跡當幀位置的那些列」關掉
            traj_cols_this = (flat_traj[:, fcur] % N)  # [B, N, L]
            B_, N_, L_ = traj_cols_this.shape
            idx_b = torch.arange(B_, device=device)[:, None, None].expand(B_, N_, L_)
            idx_n = torch.arange(N_, device=device)[None, :, None].expand(B_, N_, L_)
            same_sem_mask[idx_b, idx_n, traj_cols_this] = False

            C = sem_chunk #將h*w 切割成一塊一塊計算
            for s in range(0, N, C):
                e = min(s + C, N)
                mask_chunk = take_block(same_sem_mask, s, e)            # [B,N,C]
                mask_chunk = mask_chunk[:, None, :, None, :].expand(-1, heads, -1, 1, -1)  # [B,H,N,1,C]

                Kc = kt_h_f[:, :, s:e, :, :]   # [B,H,C,L,d_h]
                Vc = vt_h_f[:, :, s:e, :, :]   # [B,H,C,L,d_h]
                Bc, Hh, Cc, Ll, Dh = Kc.shape
                CL = Cc * Ll
                Kc_flat = Kc.reshape(Bc, Hh, CL, Dh).unsqueeze(2)# [B,H,1,CL,d_h]
                Vc_flat = Vc.reshape(Bc, Hh, CL, Dh).unsqueeze(2)# [B,H,1,CL,d_h]
                # Kc_flat = Kc.view(Bc, Hh, CL, Dh).unsqueeze(2)  
                # Vc_flat = Vc.view(Bc, Hh, CL, Dh).unsqueeze(2)  
                '''
                logits_c = torch.matmul(q_h_f * scale, Kc_flat.transpose(-2, -1))  # [B,H,N,1,CL]  同語意的全部歷史長度注意力

                mask_chunk_CL = mask_chunk.repeat_interleave(Ll, dim=-1)  # [B,H,N,1,CL]
                neg_inf = torch.finfo(logits_c.dtype).min
                logits_c = torch.where(mask_chunk_CL, logits_c, torch.full_like(logits_c, neg_inf))
'''
                logits_c = torch.matmul(q_h_f * scale, Kc_flat.transpose(-2, -1))  # [B,H,N,1,CL]

                # 對同語意歷史加時間衰減
                td_chunk = time_decay[fcur, s:e, :].reshape(1, 1, 1, 1, CL)  # [1,1,1,1,CL]
                logits_c = logits_c + torch.log(td_chunk + 1e-12)

                mask_chunk_CL = mask_chunk.repeat_interleave(Ll, dim=-1)  # [B,H,N,1,CL]
                neg_inf = torch.finfo(logits_c.dtype).min
                logits_c = torch.where(mask_chunk_CL, logits_c, torch.full_like(logits_c, neg_inf))

                m_c = torch.max(logits_c, dim=-1, keepdim=True).values     # [B,H,N,1,1]
                m_new = torch.maximum(m, m_c.squeeze(-1))                  # [B,H,N,1]
                alpha = torch.exp(m - m_new)                               # [B,H,N,1]

                Z = Z * alpha + torch.sum(torch.exp(logits_c - m_new.unsqueeze(-1)), dim=-1)  # [B,H,N,1]
                Out = Out * alpha.unsqueeze(-1) + (torch.exp(logits_c - m_new.unsqueeze(-1)) @ Vc_flat)  # [B,H,N,1,d_h]
                m = m_new

        out_f = (Out / Z.unsqueeze(-1)).squeeze(-2)  # [B,H,N,d_h]
        out_all[bf_idx] = out_f.to(base_dtype)

    hidden_states = rearrange(out_all, '(b f) h (H W) d -> b (f H W) (h d)', b=Bsmall, f=F_clip, H=H, W=W)
    return hidden_states

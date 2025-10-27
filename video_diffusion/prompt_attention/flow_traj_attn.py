# -*- coding: utf-8 -*-
import math
import torch
from einops import rearrange


# ---------------------------------------------
# Helpers
# ---------------------------------------------
def reshape_heads_to_batch_dim3(x, heads):
    """
    x: [(B*F), N, 1 or L, D] -> [(B*F), H, N, 1 or L, d_h]
    """
    Bf, N, L, D = x.shape
    assert D % heads == 0, f"D={D} must be divisible by heads={heads}"
    d_h = D // heads
    x = x.view(Bf, N, L, heads, d_h)
    x = x.permute(0, 3, 1, 2, 4).contiguous()
    return x


def take_block(mask, s, e):
    return mask[..., s:e]


def normalize_traj_and_mask(traj, mask, F_clip, N):
    """
    將輸入的 traj / mask 正規化到：
      traj: [F, N, L, 3]  (t,x,y)
      mask: [F, N, L]     (bool/int)
    支援下列形狀的自動轉換：
      - traj: [F,N,L,3] 或 [N,F,L,3] 或 [(F*N),L,3]
      - mask: [F,N,L]   或 [N,F,L]   或 [(F*N),L]
    """
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


# ---------------------------------------------
# Main: Flow + Same-Semantics Full-History Attention (No Top-K)
# ---------------------------------------------
@torch.no_grad()  # 你如果要訓練，請移除此 decorator
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

    # ---- 軌跡與 mask（均為 [F,N,L,*]）----
    t_inds = traj[..., 0].long()  # [F,N,L]
    x_inds = traj[..., 1].long()
    y_inds = traj[..., 2].long()

    # time-causal
    anchor = t_inds[:, :, 0].unsqueeze(-1).expand_as(t_inds)  # [F,N,L]
    if time_causal:
        traj_mask = t_inds <= anchor
    else:
        traj_mask = torch.ones_like(t_inds, dtype=torch.bool)
    t_inds = torch.where(traj_mask, t_inds, torch.zeros_like(t_inds))
    x_inds = torch.where(traj_mask, x_inds, torch.zeros_like(x_inds))
    y_inds = torch.where(traj_mask, y_inds, torch.zeros_like(y_inds))

    # flat 位置 id：同幀語義去重用
    flat_traj = (t_inds * N + (x_inds * W + y_inds)).unsqueeze(0).expand(Bsmall, F_clip, N, -1)  # [B,F,N,L]

    # Q
    query_tempo = query.unsqueeze(-2)  # [(B*F), N, 1, D]
    q_h = reshape_heads_to_batch_dim3(query_tempo, heads=heads).to(acc_dtype)

    # 沿軌跡 gather K/V
    key_tempo   = _key[:, t_inds, x_inds, y_inds]    # [B, F, N, L, D]
    value_tempo = _value[:, t_inds, x_inds, y_inds]  # [B, F, N, L, D]
    key_tempo   = rearrange(key_tempo,   'b f n l d -> (b f) n l d')
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
                          device=device, dtype=base_dtype)

    # sreg（full or per-frame）
    sreg_full = None
    if use_sem_aug and getattr(controller, "sreg_maps", None) is not None and controller.sreg_maps.get(h * w, None) is not None:
        sreg_full = controller.sreg_maps[h * w]  # [B or 1, FN, FN]
        assert sreg_full.dim() == 3 and sreg_full.size(1) == FN and sreg_full.size(2) == FN, \
            f"sreg shape mismatch: got {tuple(sreg_full.shape)}, expect (*,{FN},{FN})"
        if sreg_full.size(0) == 1:
            sreg_full = sreg_full.expand(Bsmall, FN, FN)
        elif sreg_full.size(0) != Bsmall:
            raise RuntimeError(f"Unexpected sreg batch dim: {sreg_full.size(0)} vs B={Bsmall}")

    # 逐幀
    for fcur in range(F_clip):
        bf_idx = (torch.arange(Bsmall, device=device) * F_clip + fcur)  # [B]
        q_h_f  = q_h[bf_idx]               # [B,H,N,1,d_h]
        kt_h_f = kt_h[bf_idx]              # [B,H,N,L,d_h]
        vt_h_f = vt_h[bf_idx]              # [B,H,N,L,d_h]

        # (1) self-traj
        logits_traj = torch.matmul(q_h_f * scale, kt_h_f.transpose(-2, -1))  # [B,H,N,1,L]
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
            q_flat = fcur * N + torch.arange(N, device=device)       # [N]
            sreg_rows = torch.take_along_dim(
                sreg_full, q_flat.view(1, -1, 1).expand(Bsmall, N, FN), dim=1
            )                                                        # [B, N, FN]
            sreg_rows_frame = sreg_rows[:, :, fcur * N:(fcur + 1) * N]   # [B, N, N]
            same_sem_mask = (sreg_rows_frame > 0)                        # [B, N, N]

            # 去掉「自身軌跡在當幀的位置」以避免重複
            traj_cols_this = (flat_traj[:, fcur] % N)  # [B, N, L]
            B_, N_, L_ = traj_cols_this.shape
            idx_b = torch.arange(B_, device=device)[:, None, None].expand(B_, N_, L_)
            idx_n = torch.arange(N_, device=device)[None, :, None].expand(B_, N_, L_)
            same_sem_mask[idx_b, idx_n, traj_cols_this] = False

            C = sem_chunk if sem_chunk > 0 else N
            for s in range(0, N, C):
                e = min(s + C, N)
                mask_chunk = take_block(same_sem_mask, s, e)            # [B,N,C]
                mask_chunk = mask_chunk[:, None, :, None, :].expand(-1, heads, -1, 1, -1)  # [B,H,N,1,C]

                Kc = kt_h_f[:, :, s:e, :, :]   # [B,H,C,L,d_h]
                Vc = vt_h_f[:, :, s:e, :, :]   # [B,H,C,L,d_h]
                Bc, Hh, Cc, Ll, Dh = Kc.shape
                CL = Cc * Ll
                Kc_flat = Kc.reshape(Bc, Hh, CL, Dh).unsqueeze(2)  # [B,H,1,CL,d_h]
                Vc_flat = Vc.reshape(Bc, Hh, CL, Dh).unsqueeze(2)  # [B,H,1,CL,d_h]

                logits_c = torch.matmul(q_h_f * scale, Kc_flat.transpose(-2, -1))  # [B,H,N,1,CL]
                mask_chunk_CL = mask_chunk.repeat_interleave(Ll, dim=-1)           # [B,H,N,1,CL]
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


# ---------------------------------------------
# sreg 相關輔助（full -> per-frame、子序列裁切）
# ---------------------------------------------
def _sreg_full_to_per_frame(sreg_full, F_clip, N, B, device):
    """
    sreg_full: [Bhz, FN, FN] (可能是 1、B 或 B*頭數)
    回傳 per-frame: [B, F, N, N] 只保留每幀的 N×N 對角塊
    """
    assert sreg_full.dim() == 3 and sreg_full.size(1) == F_clip * N and sreg_full.size(2) == F_clip * N
    Bhz, FN1, FN2 = sreg_full.shape
    assert FN1 == FN2 == F_clip * N

    if Bhz == 1:
        sreg_b = sreg_full.expand(B, FN1, FN2)
    elif Bhz == B:
        sreg_b = sreg_full
    else:
        assert Bhz % B == 0, f"Cannot map sreg batch {Bhz} to B={B}."
        Hstar = Bhz // B
        sreg_b = sreg_full.view(B, Hstar, FN1, FN2).amax(dim=1)

    sreg_b = (sreg_b > 0)

    out = torch.empty((B, F_clip, N, N), dtype=torch.bool, device=device)
    for f in range(F_clip):
        r = slice(f * N, (f + 1) * N)
        out[:, f] = sreg_b[:, r, r]
    return out


def _slice_sreg_per_frame(controller, hw, F_total, N, B, device, t0, t1):
    """
    從 controller.sreg_maps[hw] 取出 [B, F_total, N, N] 的 per-frame，再裁成子序列 [B, L_eff, N, N]
    - 支援來源是 [B or 1, F, N, N] 或 [B or 1, FN, FN]
    """
    if getattr(controller, "sreg_maps", None) is None:
        return None
    sreg_any = controller.sreg_maps.get(hw, None)
    if sreg_any is None:
        return None

    if sreg_any.dim() == 4 and sreg_any.size(1) == F_total and sreg_any.size(2) == N and sreg_any.size(3) == N:
        if sreg_any.size(0) == 1 and B > 1:
            sreg_pf = sreg_any.expand(B, F_total, N, N).contiguous()
        elif sreg_any.size(0) in (1, B):
            sreg_pf = sreg_any
        else:
            raise RuntimeError(f"sreg batch dim mismatch: {sreg_any.size(0)} vs B={B}")
        sreg_pf = (sreg_pf > 0)
    elif sreg_any.dim() == 3 and sreg_any.size(1) == F_total * N and sreg_any.size(2) == F_total * N:
        sreg_pf = _sreg_full_to_per_frame(sreg_any.to(device), F_total, N, B, device)  # [B,F,N,N]
    else:
        raise RuntimeError(f"Unsupported sreg shape {tuple(sreg_any.shape)} for F={F_total}, N={N}")

    return sreg_pf[:, t0:t1]  # [B, L_eff, N, N]


# ---------------------------------------------
# Bi-dir window wrapper（嚴格只看當前與過去）
# ---------------------------------------------
@torch.no_grad()  # 要訓練就移除此 decorator
def flow_traj_attn_bidir_window(
    query_old, key_old, value_old,
    encoder_hidden_states, group_norm,
    traj, mask,
    _key, _value,           # [B, F, H, W, D]
    h, w, clip_length, heads,
    controller, sem_chunk=128, use_sem_aug=True,
    flow_only=False, old_qk=1,
    # 視窗設定
    t=0, L=5, fuse_alpha=0.5,
):
    """
    在子序列 [t0..t]（長度 L_eff）內做「前向 causal」與「反向 causal」，融合後只取當前幀 t 的輸出。
    - 不看 t+1 之後。
    - 回傳: [B, N, D]
    """
    device = encoder_hidden_states.device
    BxF, _, D = encoder_hidden_states.shape
    H, W = h, w
    N = H * W
    B = BxF // clip_length
    assert B * clip_length == BxF

    # ---- 子序列範圍 ----
    t0 = max(0, t - (L - 1))
    t1 = t + 1
    L_eff = t1 - t0

    # ---- 正規化 → 取子序列 → 讓 t 指標變成子序列相對時間 ----
    traj_full, mask_full = normalize_traj_and_mask(traj, mask, F_clip=clip_length, N=N)  # [F,N,Ltraj,*], [F,N,Ltraj]
    traj_sub = traj_full[t0:t1].clone()
    mask_sub = mask_full[t0:t1].clone()

    # 把 (絕對 t) 轉成子序列相對 t： t' = t - t0
    t_inds = traj_sub[..., 0]
    t_inds = (t_inds - t0).clamp_(min=0, max=L_eff - 1)
    traj_sub[..., 0] = t_inds

    # ---- K/V 子序列 ----
    _key_sub   = _key[:, t0:t1]     # [B, L_eff, H, W, D]
    _value_sub = _value[:, t0:t1]   # [B, L_eff, H, W, D]

    # ---- sreg 子序列 (per-frame) ----
    sreg_pf_sub = _slice_sreg_per_frame(controller, h * w, F_total=clip_length, N=N, B=B,
                                        device=device, t0=t0, t1=t1)  # [B,L_eff,N,N] or None
    # 暫存→覆寫 controller.sreg_maps[hw] 成子序列 per-frame，方便 core 直接吃
    orig_sreg = controller.sreg_maps.get(h * w, None) if getattr(controller, "sreg_maps", None) is not None else None
    if getattr(controller, "sreg_maps", None) is not None and sreg_pf_sub is not None:
        controller.sreg_maps[h * w] = sreg_pf_sub  # core 會吃到 per-frame 版本

    # ---- 前向（子序列內）----
    out_fwd = flow_semantic_traj_attention(
        query_old=query_old, key_old=key_old, value_old=value_old,
        encoder_hidden_states=encoder_hidden_states, group_norm=group_norm,
        traj=traj_sub, mask=mask_sub, time_causal=True,
        _key=_key_sub, _value=_value_sub,
        h=h, w=w, clip_length=L_eff, heads=heads,
        controller=controller, sem_chunk=sem_chunk,
        use_sem_aug=use_sem_aug, flow_only=flow_only, old_qk=old_qk
    )  # [B, (L_eff*N), D]
    out_fwd = rearrange(out_fwd, 'b (f n) d -> b f n d', f=L_eff, n=N)  # [B, L_eff, N, D]

    # ---- 反向（子序列內翻轉 + 相對 t 反轉）----
    traj_rev = traj_sub.flip(0).clone()  # 先沿 F 維翻轉
    t_inds_rev = traj_rev[..., 0]
    t_inds_rev = (L_eff - 1) - t_inds_rev
    t_inds_rev.clamp_(min=0, max=L_eff - 1)
    traj_rev[..., 0] = t_inds_rev

    mask_rev = mask_sub.flip(0).contiguous()
    _key_rev   = _key_sub.flip(1).contiguous()
    _value_rev = _value_sub.flip(1).contiguous()
    sreg_rev = sreg_pf_sub.flip(1).contiguous() if sreg_pf_sub is not None else None
    if getattr(controller, "sreg_maps", None) is not None and sreg_rev is not None:
        controller.sreg_maps[h * w] = sreg_rev  # 覆寫成反向子序列的 per-frame

    out_bwd_rev = flow_semantic_traj_attention(
        query_old=query_old, key_old=key_old, value_old=value_old,
        encoder_hidden_states=encoder_hidden_states, group_norm=group_norm,
        traj=traj_rev, mask=mask_rev, time_causal=True,
        _key=_key_rev, _value=_value_rev,
        h=h, w=w, clip_length=L_eff, heads=heads,
        controller=controller, sem_chunk=sem_chunk,
        use_sem_aug=use_sem_aug, flow_only=flow_only, old_qk=old_qk
    )
    out_bwd_rev = rearrange(out_bwd_rev, 'b (f n) d -> b f n d', f=L_eff, n=N)  # 反向的輸出
    out_bwd = out_bwd_rev.flip(1).contiguous()  # 翻回正時間：[B, L_eff, N, D]

    # ---- 融合 & 只取當前幀 (子序列最後一幀) ----
    alpha = float(fuse_alpha)
    out_fused = alpha * out_fwd + (1.0 - alpha) * out_bwd  # [B,L_eff,N,D]
    out_t = out_fused[:, -1]  # [B, N, D] 對應原序列的幀 t

    # 還原 controller 的 sreg
    if getattr(controller, "sreg_maps", None) is not None:
        if orig_sreg is None:
            controller.sreg_maps.pop(h * w, None)
        else:
            controller.sreg_maps[h * w] = orig_sreg

    return out_t

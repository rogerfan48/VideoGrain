# video_diffusion/prompt_attention/flow_traj_attn.py
# -*- coding: utf-8 -*-
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


@torch.no_grad()  # 要訓練的話拿掉這個 decorator
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
    # ===== 新增（雙通道/雙向）=====
    bidir: bool = False,
    bidir_alpha: float = 0.5,
    bidir_L: int = None,  # 目前保留接口，這版不裁窗，最小改動
):
    """
    每個 query 既看：自己過去的軌跡（self-traj），也看同幀同語意像素及其各自的過去軌跡（same-class past-traj）。
    若 bidir=True，再做一次時間反轉的因果 pass，最後做 convex fuse: (1-α)·fwd + α·bwd。
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

    # ---- 正規化：得到 [F,N,L,*] ----
    traj, mask = normalize_traj_and_mask(traj, mask, F_clip, N)  # [F,N,L,3], [F,N,L]
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

    # flatten index：同幀語義去重用
    flat_traj = (t_inds * N + (x_inds * W + y_inds))  # [F,N,L]

    # Q
    query_tempo = query.unsqueeze(-2)  # [(B*F), N, 1, D]
    q_h = reshape_heads_to_batch_dim3(query_tempo, heads=heads).to(acc_dtype)

    # 沿軌跡 gather K/V
    with torch.autocast(device_type=str(device).split(':')[0], enabled=False):
        key_tempo   = _key[:, t_inds, x_inds, y_inds]    # [B, F, N, L, D]
        value_tempo = _value[:, t_inds, x_inds, y_inds]  # [B, F, N, L, D]
    key_tempo   = rearrange(key_tempo,   'b f n l d -> (b f) n l d')
    value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d')
    kt_h = reshape_heads_to_batch_dim3(key_tempo,   heads=heads).to(acc_dtype)  # [(B*F),H,N,L,d_h]
    vt_h = reshape_heads_to_batch_dim3(value_tempo, heads=heads).to(acc_dtype)  # [(B*F),H,N,L,d_h]

    # sreg: 支援 [B,F,N,N] 或 [Bhz,FN,FN]
    if use_sem_aug and controller.sreg_maps[h * w] is not None:
        sreg_any = controller.sreg_maps[h * w]
        if sreg_any.dim() == 4 and sreg_any.size(1) == F_clip and sreg_any.size(2) == N and sreg_any.size(3) == N:
            if sreg_any.size(0) == 1 and Bsmall > 1:
                sreg_pf = sreg_any.expand(Bsmall, F_clip, N, N).contiguous()
            else:
                sreg_pf = sreg_any
            sreg_pf = (sreg_pf > 0)
            sreg_full = None
        else:
            # 舊格式 [Bhz, FN, FN]
            sreg_full = sreg_any
            if sreg_full.size(0) == 1:
                sreg_full = sreg_full.expand(Bsmall, FN, FN)
            elif sreg_full.size(0) != Bsmall:
                raise RuntimeError(f"Unsupported sreg batch {sreg_full.size(0)} vs B={Bsmall}")
            sreg_pf = None
    else:
        sreg_pf = None
        sreg_full = None

    # ---- Streaming Softmax 容器（前向）----
    d_h = q_h.size(-1)
    scale = torch.tensor(1.0 / math.sqrt(d_h), dtype=acc_dtype, device=device)
    out_all = torch.empty((Bsmall * F_clip, heads, N, d_h), device=device, dtype=base_dtype)

    keep_mask = (mask.to(torch.bool) & traj_mask)  # [F,N,L]
    # 逐幀
    for fcur in range(F_clip):
        bf_idx = (torch.arange(Bsmall, device=device) * F_clip + fcur)   # [B]
        q_h_f  = q_h[bf_idx]    # [B,H,N,1,d_h]
        kt_f   = kt_h[bf_idx]   # [B,H,N,L,d_h]
        vt_f   = vt_h[bf_idx]   # [B,H,N,L,d_h]

        # (1) Self-traj
        logits_traj = torch.matmul(q_h_f * scale, kt_f.transpose(-2, -1))  # [B,H,N,1,L]
        keep_mask_b = keep_mask[fcur].unsqueeze(0).expand(Bsmall, N, keep_mask.size(-1))  # [B,N,L]
        attn_mask = keep_mask_b[:, None, :, None, :]  # [B,1,N,1,L]
        neg_val = torch.finfo(logits_traj.dtype).min
        logits_traj = logits_traj.masked_fill(~attn_mask, neg_val)

        m  = torch.max(logits_traj, dim=-1, keepdim=True).values  # [B,H,N,1,1]
        exp_logits = torch.exp(logits_traj - m)                   # [B,H,N,1,L]
        Z   = exp_logits.sum(dim=-1, keepdim=True)                # [B,H,N,1,1]
        Out = torch.matmul(exp_logits, vt_f)                      # [B,H,N,1,d_h]

        # (2) 同幀同語意 + 過去軌跡
        if sreg_pf is not None or sreg_full is not None:
            if sreg_pf is not None:
                same_sem_mask = sreg_pf[:, fcur]  # [B,N,N]
            else:
                # 從 full 版擷取當幀 N×N 區塊
                r = slice(fcur * N, (fcur + 1) * N)
                same_sem_mask = (sreg_full[:, r, r] > 0)  # [B,N,N]

            traj_cols_this = (flat_traj[fcur] % N)  # [N,L]
            M = torch.zeros((Bsmall, N, N), dtype=torch.bool, device=device)
            idx_scatter = traj_cols_this.unsqueeze(0).expand(Bsmall, -1, -1)  # [B,N,L]
            M.scatter_(dim=2, index=idx_scatter, src=torch.ones_like(idx_scatter, dtype=torch.bool, device=device))
            same_sem_mask = same_sem_mask & (~M)  # [B,N,N]

            Cc = sem_chunk if sem_chunk > 0 else N
            for s in range(0, N, Cc):
                e = min(s + Cc, N)
                mask_chunk = take_block(same_sem_mask, s, e)            # [B,N,C]
                mask_chunk = mask_chunk[:, None, :, None, :].expand(-1, heads, -1, 1, -1)  # [B,H,N,1,C]

                Kc = kt_f[:, :, s:e, :, :]   # [B,H,C,L,d_h]
                Vc = vt_f[:, :, s:e, :, :]   # [B,H,C,L,d_h]
                Bc, Hh, Cnum, Ll, Dh = Kc.shape
                CL = Cnum * Ll
                Kc_flat = Kc.reshape(Bc, Hh, CL, Dh).unsqueeze(2)  # [B,H,1,CL,d]
                Vc_flat = Vc.reshape(Bc, Hh, CL, Dh).unsqueeze(2)  # [B,H,1,CL,d]

                logits_c = torch.matmul(q_h_f * scale, Kc_flat.transpose(-2, -1))  # [B,H,N,1,CL]
                mask_chunk_CL = mask_chunk.repeat_interleave(Ll, dim=-1)           # [B,H,N,1,CL]
                neg_inf = torch.finfo(logits_c.dtype).min
                logits_c = torch.where(mask_chunk_CL, logits_c, torch.full_like(logits_c, neg_inf))

                m_c = torch.max(logits_c, dim=-1, keepdim=True).values     # [B,H,N,1,1]
                m_new = torch.maximum(m, m_c)                               # [B,H,N,1,1]
                alpha = torch.exp(m - m_new)                                # [B,H,N,1,1]
                Z  = Z  * alpha + torch.sum(torch.exp(logits_c - m_new), dim=-1, keepdim=True)   # [B,H,N,1,1]
                Out = Out * alpha + (torch.exp(logits_c - m_new) @ Vc_flat)                       # [B,H,N,1,d]
                m = m_new

        out_f = (Out / (Z + 1e-12)).squeeze(-2)   # [B,H,N,d_h]
        out_all[bf_idx] = out_f.to(base_dtype)

    # 回到 [(B*F), (H*W), (H*d_h)] layout
    hidden_states_fwd = rearrange(out_all, '(b f) h (H W) d -> b (f H W) (h d)', b=Bsmall, f=F_clip, H=H, W=W)

    # =====================（可選）雙向 pass =====================
    if not bidir:
        return hidden_states_fwd

    # 1) 反轉時間軸的 _key/_value
    _key_rev   = _key[:, torch.arange(F_clip-1, -1, -1, device=device), ...]
    _value_rev = _value[:, torch.arange(F_clip-1, -1, -1, device=device), ...]
    # 2) 反轉時間座標：t' = F-1 - t
    traj_rev = traj.clone()
    traj_rev[..., 0] = (F_clip - 1) - traj_rev[..., 0]
    mask_rev = mask  # mask 本身不需要數值變更（但對應的 gather 會跟著 t' 走）

    # 3) 若 sreg 是 full 版，做 FN 對應的雙側置換；per-frame 版則只需倒序 frame 索引
    if sreg_full is not None:
        # 建立 frame 反轉的 token 置換表
        perm_frames = []
        for f in range(F_clip-1, -1, -1):
            r = torch.arange(f*N, (f+1)*N, device=device)
            perm_frames.append(r)
        perm = torch.cat(perm_frames, dim=0)  # [FN]
        sreg_full_rev = sreg_full[:, perm][:, :, perm]  # [B, FN, FN]
        sreg_pf_rev = None
    elif sreg_pf is not None:
        sreg_pf_rev = sreg_pf[:, torch.arange(F_clip-1, -1, -1, device=device)]
        sreg_full_rev = None
    else:
        sreg_pf_rev = None
        sreg_full_rev = None

    # 暫時換掉 controller 的 sreg_maps
    sreg_backup = controller.sreg_maps.get(h*w, None) if hasattr(controller, "sreg_maps") else None
    if hasattr(controller, "sreg_maps"):
        if sreg_full_rev is not None:
            controller.sreg_maps[h*w] = sreg_full_rev
        elif sreg_pf_rev is not None:
            controller.sreg_maps[h*w] = sreg_pf_rev

    # 4) 重新跑一次相同邏輯（反向因果）：時間仍是 past-causal，只是資料已反轉
    # --- 下面直接「遞迴」呼叫本函式，但關掉 bidir 以避免再反向一次 ---
    hidden_states_bwd_rev = flow_semantic_traj_attention(
        query_old, key_old, value_old,
        encoder_hidden_states, group_norm,
        traj_rev, mask_rev, time_causal,
        _key_rev, _value_rev,
        h, w, clip_length, heads,
        controller, sem_chunk,
        use_sem_aug, flow_only, old_qk,
        bidir=False,  # 關掉遞迴雙向
    )

    # 還原 controller 的 sreg
    if hasattr(controller, "sreg_maps"):
        controller.sreg_maps[h*w] = sreg_backup

    # 5) 把 bwd 結果在 frame 維度反轉回原順序
    hb = rearrange(hidden_states_bwd_rev, 'b (f n) c -> b f n c', f=F_clip, n=N)
    hb = hb[:, torch.arange(F_clip-1, -1, -1, device=device)]
    hidden_states_bwd = rearrange(hb, 'b f n c -> b (f n) c')

    # 6) convex fuse
    out = (1.0 - bidir_alpha) * hidden_states_fwd + bidir_alpha * hidden_states_bwd
    return out

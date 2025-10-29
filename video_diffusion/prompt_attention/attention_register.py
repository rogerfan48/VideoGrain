"""
register the attention controller into the UNet of stable diffusion
Build a customized attention function `_attention'
Replace the original attention function with `forward' and `spatial_temporal_forward' in attention_controlled_forward function
Most of spatial_temporal_forward is directly copy from `video_diffusion/models/attention.py'
TODO FIXME: merge redundant code with attention.py
"""
from video_diffusion.prompt_attention.flow_traj_attn import (
    flow_semantic_traj_attention,
    normalize_traj_and_mask,
    reshape_heads_to_batch_dim3
)
# from video_diffusion.prompt_attention.semantic_flow_attention_high_efficiency import semantic_flow_fullframe_sreg
from einops import rearrange
import torch
import torch.nn.functional as F
import math
from diffusers.utils.import_utils import is_xformers_available
import numpy as np
import os, json, csv
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def register_attention_control(model, controller, text_cond, clip_length, height, width, ddim_inversion, id_masks_by_res=None, part_masks_by_res=None):
    "Connect a model with a controller"
    def attention_controlled_forward(self, place_in_unet, attention_type='cross'):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
####videograin _attention     
        def _attention(query, key, value, is_cross, attention_mask=None):
            if self.upcast_attention:
                query = query.float()
                key = key.float()
            # print("query",query.shape)
            # print("key",key.shape)
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )
            #print("attention_scores",attention_scores.shape)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            if self.upcast_softmax:
                attention_scores = attention_scores.float()

            # START OF CORE FUNCTION
            # if not ddim_inversion:
            attention_probs = controller(reshape_batch_dim_to_temporal_heads(attention_scores), 
                                        is_cross, place_in_unet)
            attention_probs = reshape_temporal_heads_to_batch_dim(attention_probs)
            # END OF CORE FUNCTION

            attention_probs = attention_probs.softmax(dim=-1)

            # cast back to the original dtype
            attention_probs = attention_probs.to(value.dtype)

            
            # compute attention output
            hidden_states = torch.bmm(attention_probs, value)

            # reshape hidden_states
            hidden_states = reshape_batch_dim_to_heads(hidden_states)
            return hidden_states
       
        def reshape_temporal_heads_to_batch_dim(tensor):
            head_size = self.heads
            tensor = rearrange(tensor, " b h s t -> (b h) s t ", h = head_size)
            return tensor

        def reshape_batch_dim_to_temporal_heads(tensor):
            head_size = self.heads
            tensor = rearrange(tensor, "(b h) s t -> b h s t", h = head_size)
            return tensor
        
        def reshape_heads_to_batch_dim3(tensor):
            batch_size1, batch_size2, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size1, batch_size2, seq_len, head_size, dim // head_size)
            tensor = tensor.permute(0, 3, 1, 2, 4)
            return tensor
        
        def reshape_heads_to_batch_dim(tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor

        def reshape_batch_dim_to_heads(tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
            return tensor
        def merge_heads_auto(tensor, *, B: int):
            """
            把 (B*H_used, L, Dh) 合回 (B, L, H_used*Dh)，
            用實際 B 反推出當前 attention 真正用了多少個 heads (H_used)。
            """
            bz_heads, L, Dh = tensor.shape
            if B <= 0:
                raise RuntimeError(f"[merge_heads_auto] B must be >0, got B={B}")
            if bz_heads % B != 0:
                raise RuntimeError(
                    f"[merge_heads_auto] cannot infer heads: bz_heads={bz_heads} not divisible by B={B}"
                )
            H_used = bz_heads // B
            # (B,H_used,L,Dh) -> (B,L,H_used,Dh) -> (B,L,H_used*Dh)
            return (
                tensor.reshape(B, H_used, L, Dh)
                      .transpose(1, 2)
                      .reshape(B, L, H_used * Dh)
            ), H_used

### [New]
        def _memory_efficient_attention_xformers(
                query, key, value, attention_mask, time_causal=False,
                height=None, width=None, clip_length=None,
                enable_pullpush=True,
                ref_frames=tuple(range(0, 6)),
                cross_start=6,
                ratio_thr=1.2,
                pull_strength=2.0,
                eps=1e-8,
                # [ADD] 拉扯決策模式：'qk_mean' 用子塊 logits，'qmean_kmean' 用均值近似
                decision_mode: str = "qk_mean",
                # [ADD] 分塊大小（僅對 qk_mean 有效）
                row_chunk_size: int = 4096,
                col_chunk_size: int = 8192,
            ):
            """
            在進 xFormers 之前，僅對 t >= cross_start 的該幀 S/B 行進行「方向拉扯」。
            decision_mode:
              - 'qk_mean'：以子塊 Q@K^T logits 的平均作為 μ（更準；分塊避免 OOM）
              - 'qmean_kmean'：原本的 q̄·k̄ 快速近似
            """
            import torch, xformers
            from einops import rearrange
            from xformers.ops import fmha
        
            query = query.contiguous()
            key   = key.contiguous()
            value = value.contiguous()
        
            BxH, Qlen, D = query.shape
            _,   Klen, Dk = key.shape
            assert D == Dk, f"Q/K dim mismatch: {D} vs {Dk}"
        
            device = query.device
            attn_bias = fmha.attn_bias.LowerTriangularMask() if time_causal else None
        
            # 早退：條件不足 → 直接 xFormers
            if (not enable_pullpush) or (id_masks_by_res is None) or (height is None) or (width is None) or (clip_length is None):
                out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attn_bias)
                print("no")
                return reshape_batch_dim_to_heads(out)
        
            H, W, F = height, width, clip_length
            sp_sz = H * W
            assert Qlen == F * sp_sz and Klen == F * sp_sz, "Expect F*H*W layout for Q/K"
        
            # ----- 取 ID 遮罩 -----
            def _safe_id_maps(char: str):
                if id_masks_by_res is None or (H*W) not in id_masks_by_res: return None
                mp = id_masks_by_res[H*W]
                return mp.get(char, None)
            idS = _safe_id_maps("S")
            idB = _safe_id_maps("B")
            if idS is None or idB is None:
                out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attn_bias)
                print("no")
                return reshape_batch_dim_to_heads(out)
        
            # ----- 依幀取 row/col mask -----
            def _frame_row_mask(id_maps, t):
                m = torch.zeros(Qlen, dtype=torch.bool, device=device)
                mhw = id_maps[t].to(device=device).reshape(-1).bool()
                m[t*sp_sz:(t+1)*sp_sz] = mhw
                return m
            def _frame_col_mask(id_maps, t):
                m = torch.zeros(Klen, dtype=torch.bool, device=device)
                mhw = id_maps[t].to(device=device).reshape(-1).bool()
                m[t*sp_sz:(t+1)*sp_sz] = mhw
                return m
        
            # Key 軸參考幀聯集
            S_cols_ref_union = torch.zeros(Klen, dtype=torch.bool, device=device)
            B_cols_ref_union = torch.zeros(Klen, dtype=torch.bool, device=device)
            for r in ref_frames:
                if 0 <= r < F:
                    S_cols_ref_union |= _frame_col_mask(idS, r)
                    B_cols_ref_union |= _frame_col_mask(idB, r)
            if not (S_cols_ref_union.any() and B_cols_ref_union.any()):
                out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attn_bias)
                print("no")
                return reshape_batch_dim_to_heads(out)
        
            # 參考 K 均值（方向用；決策可不依賴它）
            KS = key[:, S_cols_ref_union, :]    # [BxH, Ks, D]
            KB = key[:, B_cols_ref_union, :]    # [BxH, Kb, D]
            k_mean_S = torch.nn.functional.normalize(KS.mean(dim=1).float(), dim=-1, eps=eps).to(query.dtype)  # [BxH,D]
            k_mean_B = torch.nn.functional.normalize(KB.mean(dim=1).float(), dim=-1, eps=eps).to(query.dtype)
        
            # --- 近似決策：q̄·k̄ ---
            def _mu_via_qmean_kmean(q_rows, kS, kB):
                if q_rows.numel() == 0: return None, None
                q_mean = torch.nn.functional.normalize(q_rows.mean(dim=1).float(), dim=-1, eps=eps)  # [BxH,D]
                muS = (q_mean * kS.float()).sum(dim=-1)   # [BxH]
                muB = (q_mean * kB.float()).sum(dim=-1)   # [BxH]
                return muS, muB
        
            # --- 精準決策：子塊 logits 平均 ---
            @torch.no_grad()
            def _mu_via_logits_chunked(q_rows, k_cols_mask):
                """
                q_rows : [BxH, nQ, D]
                k_cols : key[:, mask, :] -> [BxH, nK, D]
                回傳：每個 (b,h) 的平均 logit（未除sqrtD也可；只作相對比較）
                """
                if q_rows.numel() == 0 or (not k_cols_mask.any()):
                    return None
                k_cols = key[:, k_cols_mask, :]                         # [BxH, nK, D]
                # fp32 計算更穩定
                q32 = q_rows.float()
                k32 = k_cols.float()
                sqrt_d = float(D) ** 0.5
        
                BxH_, nQ, _ = q32.shape
                _,   nK, _  = k32.shape
        
                # 分塊 Q@K^T，逐塊累加總和與計數
                total_sum = q32.new_zeros((BxH_,), dtype=torch.float32)
                total_cnt = 0
        
                for i in range(0, nQ, row_chunk_size):
                    q_blk = q32[:, i:i+row_chunk_size, :]                     # [BxH, r, D]
                    for j in range(0, nK, col_chunk_size):
                        k_blk = k32[:, j:j+col_chunk_size, :]                 # [BxH, c, D]
                        # [BxH, r, c]
                        logits_blk = torch.einsum("brd,bcd->brc", q_blk, k_blk) / sqrt_d
                        # 平均（先沿 row/col，再沿 batch-head）
                        blk_mean = logits_blk.mean(dim=(1,2))                 # [BxH]
                        total_sum += blk_mean
                        total_cnt += 1
        
                mu = total_sum / max(total_cnt, 1)                            # [BxH]
                return mu
        
            # --- 實際拉扯（fp32 計算、保持角度/尺度）---
            def _pull_rows_toward(query, row_idx, direction, alpha_per_head):
                if row_idx is None or row_idx.numel() == 0:
                    return
                q_sel = query[:, row_idx, :].float()               # [BxH, n, D]
                dir32 = torch.nn.functional.normalize(direction.float(), dim=-1, eps=eps)  # [BxH,D]
                # head-wise 強度：alpha_per_head shape [BxH]，broadcast 到 [BxH, n, D]
                q_sel = q_sel + dir32.unsqueeze(1) * alpha_per_head.unsqueeze(-1).unsqueeze(1)
                # （可選）把長度恢復回原來均值，避免只拉長度不改角度
                base_norm = query[:, row_idx, :].float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
                q_sel = torch.nn.functional.normalize(q_sel, dim=-1, eps=eps) * base_norm
                query[:, row_idx, :] = q_sel.to(query.dtype)
        
            # --- 幀迴圈：決策 + 拉 ---
            for t in range(cross_start, F):
                mS_row = _frame_row_mask(idS, t)
                mB_row = _frame_row_mask(idB, t)
                idxS = mS_row.nonzero(as_tuple=False).squeeze(-1) if mS_row.any() else None
                idxB = mB_row.nonzero(as_tuple=False).squeeze(-1) if mB_row.any() else None
        
                # 決策使用哪種 μ
                def decide_and_pull(idx_rows, good_cols_mask, bad_cols_mask, good_kmean, bad_kmean, good_minus_bad):
                    if idx_rows is None or idx_rows.numel() == 0:
                        return
                    q_rows = query[:, idx_rows, :]  # [BxH, n, D]
        
                    if decision_mode == "qk_mean":
                        mu_good = _mu_via_logits_chunked(q_rows, good_cols_mask)   # [BxH] or None
                        mu_bad  = _mu_via_logits_chunked(q_rows, bad_cols_mask)    # [BxH] or None
                    else:
                        mu_good, mu_bad = _mu_via_qmean_kmean(q_rows, good_kmean, bad_kmean)
        
                    if (mu_good is None) or (mu_bad is None):
                        return
        
                    # 觸發條件：錯誤分數高於正確分數 * ratio_thr
                    # 這裡「錯誤」定義為 bad，相對「正確」為 good
                    cond = (mu_bad > (mu_good * ratio_thr))    # [BxH] 布林
                    if not cond.any():
                        print("no cond")
                        return
                    print("ssswap")
                    # head-wise 強度：與錯誤程度成比例
                    gap = (mu_bad / (mu_good * ratio_thr + 1e-8)) - 1.0   # [BxH], >0 表示越錯越多
                    alpha_eff = pull_strength * torch.clamp(gap, min=0.0, max=1.0)  # [BxH]
                    # 只對命中的 head 生效
                    alpha_eff = torch.where(cond, alpha_eff, alpha_eff.new_zeros(alpha_eff.shape))
        
                    direction = good_minus_bad   # [BxH,D]，例如 (k_mean_S - k_mean_B)
                    _pull_rows_toward(query, idx_rows, direction, alpha_eff)
        
                # S 行：若 S→B 大於 S→S * ratio_thr，往 (kS - kB) 拉
                decide_and_pull(
                    idxS, S_cols_ref_union, B_cols_ref_union,
                    k_mean_S, k_mean_B,
                    good_minus_bad=(k_mean_S - k_mean_B)
                )
                # B 行：若 B→S 大於 B→B * ratio_thr，往 (kB - kS) 拉
                decide_and_pull(
                    idxB, B_cols_ref_union, S_cols_ref_union,
                    k_mean_B, k_mean_S,
                    good_minus_bad=(k_mean_B - k_mean_S)
                )
        
            # ---- 進 xFormers ----
            out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attn_bias)
            return reshape_batch_dim_to_heads(out)
    
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            # hidden_states: torch.Size([16, 4096, 320])
            # encoder_hidden_states: torch.Size([16, 77, 768])
            # print("========================= Cross Attentoin ===============================")
            is_cross = encoder_hidden_states is not None
            
            #encoder_hidden_states = encoder_hidden_states

            text_cond_frames = text_cond.repeat_interleave(clip_length, 0)     # wrong implementation text_cond.repeat(clip_length,1,1)


            encoder_hidden_states = text_cond_frames

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            query = reshape_heads_to_batch_dim(query)

            if self.added_kv_proj_dim is not None:
                key = self.to_k(hidden_states)
                value = self.to_v(hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                key = reshape_heads_to_batch_dim(key)
                value = reshape_heads_to_batch_dim(value)
                encoder_hidden_states_key_proj = reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
                encoder_hidden_states_value_proj = reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

                key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
                value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
            else:
                encoder_hidden_states = text_cond_frames if encoder_hidden_states is not None else hidden_states
                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                key = reshape_heads_to_batch_dim(key)
                value = reshape_heads_to_batch_dim(value)

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            if self._use_memory_efficient_attention_xformers and query.shape[-2] > ((height//2) * (width//2)):
                hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask)
            
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
                # print("_attention")
                hidden_states = _attention(query, key, value, is_cross=is_cross, attention_mask=attention_mask)
                # hidden_states = _attention(query, key, value, attention_mask=attention_mask)
                # else:
                #     hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            #dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states


        def spatial_temporal_forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            clip_length: int = None,
            SparseCausalAttention_index: list = [-1, 'first']  #list = [0]
        ):
            # print("==============Sparse Causal Attention =====================")
            """
            Most of spatial_temporal_forward is directly copy from `video_diffusion.models.attention.SparseCausalAttention'
            We add two modification
            1. use self defined attention function that is controlled by AttentionControlEdit module
            2. remove the dropout to reduce randomness
            FIXME: merge redundant code with attention.py

            """
            if (
                self.added_kv_proj_dim is not None
                or encoder_hidden_states is not None
                or attention_mask is not None
            ):
                raise NotImplementedError

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            query = reshape_heads_to_batch_dim(query)


            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            if clip_length is not None:
                key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
                value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)


                #  *********************** Start of Spatial-temporal attention **********
                frame_index_list = []
                
                if len(SparseCausalAttention_index) > 0:
                    for index in SparseCausalAttention_index:
                        if isinstance(index, str):
                            if index == 'first':
                                frame_index = [0] * clip_length
                            if index == 'last':
                                frame_index = [clip_length-1] * clip_length
                            if (index == 'mid') or (index == 'middle'):
                                frame_index = [int((clip_length-1)//2)] * clip_length
                        else:
                            assert isinstance(index, int), 'relative index must be int'
                            frame_index = torch.arange(clip_length) + index
                            frame_index = frame_index.clip(0, clip_length-1)
                            
                        frame_index_list.append(frame_index)
                    # print("frame_index_list",frame_index_list)   [bz, frame, 4096, 320]

                    key = torch.cat([   key[:, frame_index] for frame_index in frame_index_list   #[bz, frame, 8192, 320])
                                        ], dim=2)
                    value = torch.cat([ value[:, frame_index] for frame_index in frame_index_list
                                        ], dim=2)

                
                #  *********************** End of Spatial-temporal attention **********
                key = rearrange(key, "b f d c -> (b f) d c", f=clip_length)
                value = rearrange(value, "b f d c -> (b f) d c", f=clip_length)
                # print("key after rearrange",key.shape)
                # print("value after rearrange",value.shape)

            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            # print("query after head to batch dim",query.shape)
            # print("key after head to batch dim",key.shape)

            if torch.isnan(query.reshape(-1)[0]): 
                print("nan value query",query.reshape(-1)[:10])
                print("nan value key",key.reshape(-1)[:10])
                exit()

            # print("query after reshape heads to batch ",query.shape)
            # print("key after reshape heads to batch",key.shape)

            if self._use_memory_efficient_attention_xformers and query.shape[-2] > ((height//2) * (width//2)):
                # FIXME there should be only one variable to control whether use xformers
                # if self._use_memory_efficient_attention_xformers:
                # for large attention map of 64X64, use xformers to save memory
                hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
            # if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                # hidden_states = _attention(query, key, value, attention_mask=attention_mask, is_cross=False)
                hidden_states = _attention(query, key, value, attention_mask=attention_mask)
            # else:
            #     hidden_states = self._sliced_attention(
            #         query, key, value, hidden_states.shape[1], dim, attention_mask
            #     )

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        def build_frame_causal_mask(sequence_length, clip_length, device, dtype,
                                    include_same_frame=True):
            """
            回傳形狀為 (1, Q, K) 的 additive mask，允許位置=0，禁止位置=-inf
            Q=K=sequence_length=t*hw
            include_same_frame=True  -> 允許同一幀內互看 (<= 幀下三角)
            include_same_frame=False -> 僅允許過去幀 (< 幀嚴格下三角)
            """
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            t = clip_length
            assert sequence_length % t == 0, "sequence_length 必須能被 clip_length 整除"
            hw = sequence_length // t
        
            # 幀層級下三角 (t x t)
            time_tri = torch.tril(torch.ones(t, t, device=device, dtype=torch.bool),
                                  diagonal=0 if include_same_frame else -1)  # <= or <
            # 將每個幀格放大成 hw x hw 的 block（Kronecker 乘積）
            block = torch.ones(hw, hw, device=device, dtype=torch.bool)
            frame_block_mask = torch.kron(time_tri, block)  # (t*hw, t*hw) = (Q,K)
        
            # 轉成 additive mask：允許=0，禁止=-inf
            finfo = torch.finfo(torch.float32 if dtype == torch.float16 else dtype)
            additive = torch.where(frame_block_mask, torch.zeros(1, device=device, dtype=torch.float32),
                                   torch.full((1,), finfo.min, device=device, dtype=torch.float32))
            # 形狀對齊到 (1, Q, K) 以利 broadcast 到 (B*H, Q, K)
            additive = additive.view(1, sequence_length, sequence_length)
        
            # 若你啟用 upcast_softmax，把 mask 也用 float32，避免精度問題
            return additive
    

###Great!!!!
        def _sliced_attention(query, key, value, sequence_length, dim, attention_mask, time_causal,
                              height=None, width=None, clip_length=None):
            enable_record = True
            # print("slic")
            #query (bz*heads, t x h x w, org_dim//heads )
            is_cross = False
            batch_size_attention = query.shape[0]  # bz * heads
            hidden_states = torch.zeros(
                (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
            )
        
            slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        
            if ddim_inversion:
                per_frame_len = sequence_length // clip_length
                attention_store = torch.zeros(
                    (batch_size_attention, clip_length, per_frame_len, per_frame_len),
                    device=query.device,
                    dtype=query.dtype,
                )
        
            # ===================== 基本參數/緩衝 =====================
            H, W, F = height, width, clip_length
            sp_sz = H * W
            assert sequence_length == F * sp_sz, "Q/K 長度必須等於 F*H*W"
        
            device = query.device
            dtype = query.dtype
            layer_name = place_in_unet.replace("/", "_") if isinstance(place_in_unet, str) else str(place_in_unet)
################################################################################################              
            ##### 你要的設定：參考幀與交錯起點
            REF_FRAMES = list(range(0, 6))  # 0..5 幀
            CROSS_START = 10  # 從第 6 幀開始視為交錯段
            record_after_softmax = True  # 統計 old/new 時是否記機率（維持你的原習慣）
    
            ###交換與縮放強度
            EXCHANGE_STRENGTH = 2.0             # 均值交換強度（維持你原本邏輯）
            SHRINK_FACTOR    = 0.5     # <<<< [ADD] 縮小跨類注意力（StoB / BtoS）之倍率（0~1）
            SHRINK_ON_TAIL   = True        # <<<< [ADD] 僅在發現 "跨類 > 同類" 時才縮
        
            # ------------------ [MOD] 交換條件：逐 head/逐 slice-row 判斷 ------------------
            base_ratio_min = 1.00    # 最低相對倍率門檻
            delta_thr = 0.25          # 比 baseline 至少高 25% 才觸發
            eps = 1e-6
            max_heads_to_fix = max(1, self.heads // 2)
################################################################################################       
            # 全域/分層緩衝（保留你原本的）
            if not hasattr(self, "_ap_pairs"):
                self._ap_pairs = ["S_to_S", "S_to_B", "B_to_S", "B_to_B"]
            if not hasattr(self, "_ap_layer_stats"):
                self._ap_layer_stats = {}
            if not hasattr(self, "_ap_global_sum"):
                self._ap_global_sum = torch.zeros(4, dtype=torch.float32, device=device)
            if not hasattr(self, "_ap_global_cnt"):
                self._ap_global_cnt = torch.zeros(4, dtype=torch.float32, device=device)
            if layer_name not in self._ap_layer_stats:
                self._ap_layer_stats[layer_name] = {"sum": None, "cnt": None}
        
            # ============= S/B 的 per-frame mask 構建工具（沿用你原本的解析度索引） =============
            def _frame_row_mask(id_char: str, frame_idx: int) -> torch.Tensor:
                """Query 維度的 row mask：只開啟 frame_idx 這一幀且屬於 id_char 的像素"""
                id_maps = id_masks_by_res[H * W][id_char]  # [F,H,W]
                m = torch.zeros(sequence_length, dtype=torch.bool, device=device)
                mask_hw = id_maps[frame_idx].to(device=device).reshape(-1).bool()
                start = frame_idx * sp_sz
                m[start : start + sp_sz] = mask_hw
                return m
        
            def _frame_col_mask(id_char: str, frame_idx: int) -> torch.Tensor:
                """Key 維度的 col mask：只開啟 frame_idx 這一幀且屬於 id_char 的像素"""
                id_maps = id_masks_by_res[H * W][id_char]  # [F,H,W]
                m = torch.zeros(sequence_length, dtype=torch.bool, device=device)
                mask_hw = id_maps[frame_idx].to(device=device).reshape(-1).bool()
                start = frame_idx * sp_sz
                m[start : start + sp_sz] = mask_hw
                return m
        
            try:
                S_cols_ref_union = torch.zeros(sequence_length, dtype=torch.bool, device=device)
                B_cols_ref_union = torch.zeros(sequence_length, dtype=torch.bool, device=device)
                for r in REF_FRAMES:
                    S_cols_ref_union |= _frame_col_mask("S", r)
                    B_cols_ref_union |= _frame_col_mask("B", r)
                pair_to_idx = {p: i for i, p in enumerate(self._ap_pairs)}
            except Exception as e:
                print("[AP] build ref masks failed, skip exchange:", e)
                S_cols_ref_union = B_cols_ref_union = None
        
            # ==== NEW: 初始化 / 蒐集工具（同層、同 head 聚合；支援 old/new 兩組） ====
            def _ap_init_if_needed(which: str, num_heads: int):
                if not hasattr(self, "_ap_pairs"):
                    self._ap_pairs = ["S_to_S", "S_to_B", "B_to_S", "B_to_B"]
                if not hasattr(self, "_ap_stats"):
                    self._ap_stats = {}
                if which not in self._ap_stats:
                    self._ap_stats[which] = {
                        "layer": {},
                        "g_sum": torch.zeros(4, dtype=torch.float32, device=device),
                        "g_cnt": torch.zeros(4, dtype=torch.float32, device=device),
                    }
                lay = self._ap_stats[which]["layer"]
                if layer_name not in lay or lay[layer_name]["sum"] is None:
                    lay[layer_name] = {
                        "sum": torch.zeros((self.heads, 4), dtype=torch.float32, device=device),
                        "cnt": torch.zeros((self.heads, 4), dtype=torch.float32, device=device),
                    }
        
            def _ap_collect(
                A_base: torch.Tensor,
                start_idx: int,
                end_idx: int,
                num_heads: int,
                which: str,
                pair_masks_dict: dict,
                enable_record: bool,
            ):
                """
                A_base: [slice_size, Q, K]（logits 或 prob）
                pair_masks_dict: {"S_to_S":(row_mask, col_mask), ...}
                這裡允許每次給不同的 row/col（例如每個 t）
                """
                if not enable_record:
                    return
                _ap_init_if_needed(which, num_heads)
                lay = self._ap_stats[which]["layer"][layer_name]
                g_sum = self._ap_stats[which]["g_sum"]
                g_cnt = self._ap_stats[which]["g_cnt"]
        
                global_rows = torch.arange(start_idx, end_idx, device=device)  # [slice_size]
                for local_j in range(A_base.shape[0]):
                    global_idx = int(global_rows[local_j].item())
                    head_id = global_idx % num_heads
                    A = A_base[local_j]  # [Q,K]
                    for pair_name, (row_mask, col_mask) in pair_masks_dict.items():
                        if row_mask is None or col_mask is None:
                            continue
                        sub = A[row_mask][:, col_mask]  # [nq, nk]
                        if sub.numel() > 0:
                            v = sub.mean()
                            idx = pair_to_idx[pair_name]
                            lay["sum"][head_id, idx] += v.detach().float()
                            lay["cnt"][head_id, idx] += 1.0
                            g_sum[idx] += v.detach().float()
                            g_cnt[idx] += 1.0
        
            def _maybe_prob(A: torch.Tensor):
                return torch.softmax(A, dim=-1) if record_after_softmax else A
        
            # ======= 交換工具（logits 階段） =======
            def _mk3(mask_row: torch.Tensor, mask_col: torch.Tensor, slice_len: int, Q: int, K: int):
                Mr = mask_row.unsqueeze(1) & torch.ones(K, dtype=torch.bool, device=mask_row.device)  # [Q,K]
                Mc = torch.ones(Q, dtype=torch.bool, device=mask_col.device).unsqueeze(1) & mask_col.unsqueeze(0)  # [Q,K]
                M = Mr & Mc
                return M.unsqueeze(0).expand(slice_len, -1, -1)  # [slice, Q, K]
        
            def _exchange_by_mean_shift(
                attn_logits: torch.Tensor,
                row_mask: torch.Tensor,
                col_mask_a: torch.Tensor,
                col_mask_b: torch.Tensor,
                strength: float = 1.0,
                eps: float = 1e-8,
            ):
                """
                對 attn_logits 的兩個子區塊 A(row_mask,col_mask_a) 與 B(row_mask,col_mask_b) 進行「均值交換/靠攏」：
                A <- A + strength * (μ_B - μ_A)
                B <- B - strength * (μ_B - μ_A)
                """
                slice_len, Q, K = attn_logits.shape
                Ma = _mk3(row_mask, col_mask_a, slice_len, Q, K)
                Mb = _mk3(row_mask, col_mask_b, slice_len, Q, K)
                Ma_f = Ma.float()
                Mb_f = Mb.float()
        
                sum_a = (attn_logits * Ma_f).sum(dim=-1)  # [slice, Q]
                cnt_a = Ma_f.sum(dim=-1).clamp_min(eps)   # [slice, Q]
                mu_a = sum_a / cnt_a
        
                sum_b = (attn_logits * Mb_f).sum(dim=-1)  # [slice, Q]
                cnt_b = Mb_f.sum(dim=-1).clamp_min(eps)   # [slice, Q]
                mu_b = sum_b / cnt_b
        
                row_mask_Q = row_mask.unsqueeze(0).expand(slice_len, -1)  # [slice, Q]
                delta = (mu_b - mu_a) * strength
                delta = torch.where(row_mask_Q, delta, torch.zeros_like(delta))
        
                # 把 delta 寫回兩塊（逐元素）
                attn_logits[Ma] = (attn_logits[Ma] + delta.unsqueeze(-1).expand_as(attn_logits).masked_select(Ma)).reshape(-1)
                attn_logits[Mb] = (attn_logits[Mb] - delta.unsqueeze(-1).expand_as(attn_logits).masked_select(Mb)).reshape(-1)
        
            # <<<< [ADD] 跨類縮放：把特定子塊乘上一個 <1 的係數，以縮小 StoB / BtoS 注意力 >>>>
            def _shrink_block(attn_logits: torch.Tensor, row_mask: torch.Tensor, col_mask: torch.Tensor, factor: float):
                """
                僅縮放 row_mask × col_mask 子塊： attn[row,col] *= factor
                - 建議 factor ∈ (0,1]；越小縮得越多
                """
                if factor >= 1.0:
                    return
                slice_len, Q, K = attn_logits.shape
                M = _mk3(row_mask, col_mask, slice_len, Q, K)
                # 直接就地乘
                attn_logits[M] = (attn_logits[M] * factor)
        
            # [NEW] 回傳「逐 local_j（= bz×head 切片）」的 pair means：shape = [slice_size, 4]
            def _pair_means_for_t_perhead(attn_logits: torch.Tensor, t: int):
                if S_cols_ref_union is None:
                    return None, None, None
                S_row_t = _frame_row_mask("S", t)
                B_row_t = _frame_row_mask("B", t)
                slice_len = attn_logits.shape[0]
                out = torch.full((slice_len, 4), float("nan"), device=device, dtype=attn_logits.dtype)
                for j in range(slice_len):
                    A = attn_logits[j]  # [Q,K]
                    # S row
                    ss = A[S_row_t][:, S_cols_ref_union]
                    sb = A[S_row_t][:, B_cols_ref_union]
                    # B row
                    bs = A[B_row_t][:, S_cols_ref_union]
                    bb = A[B_row_t][:, B_cols_ref_union]
                    out[j, 0] = ss.mean() if ss.numel() > 0 else float("nan")  # S_to_S
                    out[j, 1] = sb.mean() if sb.numel() > 0 else float("nan")  # S_to_B
                    out[j, 2] = bs.mean() if bs.numel() > 0 else float("nan")  # B_to_S
                    out[j, 3] = bb.mean() if bb.numel() > 0 else float("nan")  # B_to_B
                return out, S_row_t, B_row_t
        
            # baseline（避免天生跨類的 head 被誤判）
            def _compute_baseline_ratios_perhead(attn_logits: torch.Tensor):
                if S_cols_ref_union is None:
                    return None, None
                sb_over_ss = []
                bs_over_bb = []
                eps = 1e-6
                for t in REF_FRAMES:
                    means_j, _, _ = _pair_means_for_t_perhead(attn_logits, t)
                    if means_j is None:
                        continue
                    ss = means_j[:, 0]; sb = means_j[:, 1]; bs = means_j[:, 2]; bb = means_j[:, 3]
                    sb_over_ss.append( (sb / (ss.abs()+eps)) )
                    bs_over_bb.append( (bs / (bb.abs()+eps)) )
                if len(sb_over_ss) == 0:
                    return None, None
                baseline_sb_ss = torch.stack(sb_over_ss, dim=0).nanmean(dim=0)  # [slice_len]
                baseline_bs_bb = torch.stack(bs_over_bb, dim=0).nanmean(dim=0)
                return baseline_sb_ss, baseline_bs_bb
        
            # ===================== 主迴圈 =====================
            for i in range(hidden_states.shape[0] // slice_size):
                start_idx = i * slice_size
                end_idx = (i + 1) * slice_size
        
                query_slice = query[start_idx:end_idx]
                key_slice = key[start_idx:end_idx]
        
                if self.upcast_attention:
                    query_slice = query_slice.float()
                    key_slice = key_slice.float()
        
                attn_slice = torch.baddbmm(
                    torch.empty(
                        slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device
                    ),
                    query_slice,
                    key_slice.transpose(-1, -2),
                    beta=0,
                    alpha=self.scale,
                )
        
                if attention_mask is not None:
                    if time_causal:
                        attn_slice = attn_slice + attention_mask[:1]
                    else:
                        attn_slice = attn_slice + attention_mask[start_idx:end_idx]
        
                if self.upcast_softmax:
                    attn_slice = attn_slice.float()
        
                attn_slice_old = attn_slice.clone().detach()  # 交換前的快照 (old, logits)
        
                # baseline（逐 local_j/head）
                baseline_sb_ss, baseline_bs_bb = _compute_baseline_ratios_perhead(attn_slice_old)

                
                if (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
                    for t in range(CROSS_START, F):
                        means_j, S_row_t, B_row_t = _pair_means_for_t_perhead(attn_slice, t)
                        if means_j is None:
                            continue
                        ss = means_j[:, 0]; sb = means_j[:, 1]; bs = means_j[:, 2]; bb = means_j[:, 3]
                        # print("[CHK] slice_len:", means_j.shape[0], " slice_size=", slice_size)
                        now_sb_ss = sb / (ss.abs() + eps)
                        now_bs_bb = bs / (bb.abs() + eps)
        
                        thr_sb_ss = (baseline_sb_ss * (1.0 + delta_thr)) if baseline_sb_ss is not None else torch.full_like(now_sb_ss, base_ratio_min)
                        thr_bs_bb = (baseline_bs_bb * (1.0 + delta_thr)) if baseline_bs_bb is not None else torch.full_like(now_bs_bb, base_ratio_min)
        
                        viol_S = (now_sb_ss > torch.maximum(thr_sb_ss, torch.full_like(thr_sb_ss, base_ratio_min)))
                        viol_B = (now_bs_bb > torch.maximum(thr_bs_bb, torch.full_like(thr_bs_bb, base_ratio_min)))
        
                        score_S = (now_sb_ss - thr_sb_ss).nan_to_num(0.0) if baseline_sb_ss is not None else (now_sb_ss - base_ratio_min).nan_to_num(0.0)
                        score_B = (now_bs_bb - thr_bs_bb).nan_to_num(0.0) if baseline_bs_bb is not None else (now_bs_bb - base_ratio_min).nan_to_num(0.0)
        
                        idx_S = torch.nonzero(viol_S, as_tuple=False).flatten()
                        idx_B = torch.nonzero(viol_B, as_tuple=False).flatten()
        
                        # ===== 第一步：先做「均值交換」把注意力拉回同類 =====
                        if idx_S.numel() > 0:
                            topk = min(max_heads_to_fix, idx_S.numel())
                            # print(f"max head to fix:{max_heads_to_fix}, idxs:{idx_S.numel()}, topk:{topk}")
                            topS = idx_S[torch.topk(score_S[idx_S], k=topk, largest=True).indices]
                            for j in topS:
                                _exchange_by_mean_shift(
                                    attn_slice[j:j+1],
                                    row_mask=S_row_t,
                                    col_mask_a=S_cols_ref_union,  # 正確（StoS）
                                    col_mask_b=B_cols_ref_union,  # 錯誤（StoB）
                                    strength=EXCHANGE_STRENGTH,
                                    eps=eps,
                                )
        
                        if idx_B.numel() > 0:
                            topk = min(max_heads_to_fix, idx_B.numel())
                            # print(f"max head to fix:{max_heads_to_fix}, idxs:{idx_B.numel()}, topk:{topk}")
                            topB = idx_B[torch.topk(score_B[idx_B], k=topk, largest=True).indices]
                            for j in topB:
                                _exchange_by_mean_shift(
                                    attn_slice[j:j+1],
                                    row_mask=B_row_t,
                                    col_mask_a=B_cols_ref_union,  # 正確（BtoB）
                                    col_mask_b=S_cols_ref_union,  # 錯誤（BtoS）
                                    strength=EXCHANGE_STRENGTH,
                                    eps=eps,
                                )
        
                        # ===== 第二步 [CHANGE]：若「交換後已成功」（同類 > 跨類），才縮小跨類權重 =====
                        if SHRINK_ON_TAIL and (idx_S.numel() > 0 or idx_B.numel() > 0):
                            means_after, _, _ = _pair_means_for_t_perhead(attn_slice, t)
                            if means_after is not None:
                                ss2 = means_after[:, 0]; sb2 = means_after[:, 1]
                                bs2 = means_after[:, 2]; bb2 = means_after[:, 3]
        
                                # 成功條件：StoS > StoB、BtoB > BtoS
                                success_S = (ss2 > sb2)  # S 行成功
                                success_B = (bb2 > bs2)  # B 行成功
        
                                # 只在「成功」的 head 上進一步縮小跨類注意力
                                idx_success_S = torch.nonzero(success_S, as_tuple=False).flatten()
                                idx_success_B = torch.nonzero(success_B, as_tuple=False).flatten()
        
                                for j in idx_success_S:
                                    # 抑制 S row × (B cols) -> 縮 StoB
                                    # print("shrink")
                                    _shrink_block(attn_slice[j:j+1], S_row_t, B_cols_ref_union, factor=SHRINK_FACTOR)
        
                                for j in idx_success_B:
                                    # 抑制 B row × (S cols) -> 縮 BtoS
                                    # print("shrink")
                                    _shrink_block(attn_slice[j:j+1], B_row_t, S_cols_ref_union, factor=SHRINK_FACTOR)

        
                attn_slice_exch = attn_slice.clone().detach()  # 交換/縮放後、controller 前的快照 (exch, logits)
        
                # ------------------ 統計蒐集：old / exch 放在 t 迴圈內逐幀收 ------------------
                if (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
                    num_heads = self.heads
                    for t in range(CROSS_START, F):
                        try:
                            S_row_t = _frame_row_mask("S", t)
                            B_row_t = _frame_row_mask("B", t)
                        except KeyError:
                            continue
                        pm = {
                            "S_to_S": (S_row_t, S_cols_ref_union),
                            "S_to_B": (S_row_t, B_cols_ref_union),
                            "B_to_S": (B_row_t, S_cols_ref_union),
                            "B_to_B": (B_row_t, B_cols_ref_union),
                        }
                        if attn_slice_old is not None:
                            _ap_collect(
                                _maybe_prob(attn_slice_old),
                                start_idx,
                                end_idx,
                                num_heads,
                                which="old",
                                pair_masks_dict=pm,
                                enable_record=enable_record,
                            )
                        _ap_collect(
                            _maybe_prob(attn_slice_exch),
                            start_idx,
                            end_idx,
                            num_heads,
                            which="exch",
                            pair_masks_dict=pm,
                            enable_record=enable_record,
                        )
        
                # ------------------ controller ------------------
                controller_applied = False
                if i < self.heads and not ddim_inversion:
                    attention_out = controller(attn_slice.unsqueeze(1), is_cross, place_in_unet, height, width, clip_length)
                    attn_slice = attention_out.squeeze(1)
                    controller_applied = True
        
                # 若要記錄 "new"（controller 後）也可以：
                if (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
                    num_heads = self.heads
                    for t in range(CROSS_START, F):
                        try:
                            S_row_t = _frame_row_mask("S", t)
                            B_row_t = _frame_row_mask("B", t)
                        except KeyError:
                            continue
                        pm = {
                            "S_to_S": (S_row_t, S_cols_ref_union),
                            "S_to_B": (S_row_t, B_cols_ref_union),
                            "B_to_S": (B_row_t, S_cols_ref_union),
                            "B_to_B": (B_row_t, B_cols_ref_union),
                        }
                        _ap_collect(
                            _maybe_prob(attn_slice),
                            start_idx,
                            end_idx,
                            num_heads,
                            which="new",
                            pair_masks_dict=pm,
                            enable_record=enable_record,
                        )
        
                attn_slice = attn_slice.softmax(dim=-1)
                attn_slice = attn_slice.to(value.dtype)
        
                if ddim_inversion:
                    bz, thw, thw = attn_slice.shape
                    t = clip_length
                    hw = thw // t
                    per_frame_attention = torch.empty((bz, t, hw, hw), device=attn_slice.device)
                    for idx in range(t):
                        start_idx_ = idx * hw
                        end_idx_ = (idx + 1) * hw
                        per_frame_attention[:, idx, :, :] = attn_slice[:, start_idx_:end_idx_, start_idx_:end_idx_]
                    per_frame_attention = rearrange(per_frame_attention, "b t h w -> (b t) h w")
                    attention_store[start_idx:end_idx] = per_frame_attention
        
                attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
                hidden_states[start_idx:end_idx] = attn_slice
        
            if ddim_inversion:
                _ = controller(attention_store, is_cross, place_in_unet)
        
            # reshape hidden_states
            hidden_states = reshape_batch_dim_to_heads(hidden_states)    

        #     # ===================== 每次呼叫：輸出 old/new 統計 =====================
            if enable_record:
                # print("record")
                try:
                    outdir = "./result/Roger/*final_result/attn_record/3cls_10frame"
                    os.makedirs(outdir, exist_ok=True)
        
                    def _write_layer(which: str, suffix: str):
                        if not hasattr(self, "_ap_stats") or which not in self._ap_stats:
                            return
                        laydict = self._ap_stats[which]["layer"]
                        if layer_name not in laydict:
                            return
                        stat = laydict[layer_name]
                        s = stat["sum"].detach().float().cpu()
                        c = stat["cnt"].detach().float().cpu()
                        with torch.no_grad():
                            mask = c > 0
                            mean = torch.zeros_like(s)
                            mean[mask] = s[mask] / torch.clamp(c[mask], min=1e-8)
                        csv_path = os.path.join(outdir, f"{layer_name}_pairs_0to5_{suffix}.csv")  # [MOD] 檔名說明 0..5 參考
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(["head", "S_to_S", "S_to_B", "B_to_S", "B_to_B", "head_mean_over_pairs"])
                            for h in range(mean.shape[0]):
                                vals = mean[h].tolist()
                                head_mean = float(sum(vals) / len(vals)) if len(vals) > 0 else float("nan")
                                writer.writerow([h] + [float(x) for x in vals] + [head_mean])
                        torch.save(
                            {"mean": mean, "sum": s, "cnt": c},
                            os.path.join(outdir, f"{layer_name}_pairs_0to5_{suffix}.pt"),
                        )
                        with open(os.path.join(outdir, f"{layer_name}_pairs_0to5_{suffix}.json"), "w", encoding="utf-8") as jf:
                            json.dump({"mean": mean.tolist()}, jf, ensure_ascii=False, indent=2)
        
                    def _write_global(which: str, suffix: str):
                        if not hasattr(self, "_ap_stats") or which not in self._ap_stats:
                            return
                        g_sum = self._ap_stats[which]["g_sum"].detach().float().cpu()
                        g_cnt = self._ap_stats[which]["g_cnt"].detach().float().cpu()
                        if g_cnt.sum().item() > 0:
                            g_mean = g_sum / torch.clamp(g_cnt, min=1e-8)
                            glb = {p: float(g_mean[i].item()) for i, p in enumerate(self._ap_pairs)}
                        else:
                            glb = {p: None for p in self._ap_pairs}
                        with open(os.path.join(outdir, f"GLOBAL_pairs_0to5_{suffix}.json"), "w", encoding="utf-8") as f:
                            json.dump(glb, f, ensure_ascii=False, indent=2)
        
                    _write_layer("old", "old")
                    _write_layer("new", "new")
                    _write_global("old", "old")
                    _write_global("new", "new")
        
                    # === [ADD] 把 exch 一起寫出 ===
                    _write_layer("exch", "exch")  # <== 新增
                    _write_global("exch", "exch")  # <== 新增
                except Exception as e:
                    print(f"[AttnPairRecorder] write-out failed: {e}")
        
            return hidden_states


### original

        def fully_frame_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, clip_length=None, inter_frame=False, flow_only=True, time_causal=True, **kwargs):
            # print(" ====== attn1 is displayed by attention register (attention_register.py line 377) ======")
            # print(encoder_hidden_states) # None
            
            batch_size, sequence_length, _ = hidden_states.shape
            # print("hidden_states.shape",hidden_states.shape)
            # print("sequence_length",sequence_length)
            # print("======================== Full Frame forward ========================")
            encoder_hidden_states = encoder_hidden_states
            h = kwargs['height']
            w = kwargs['width']
#############################################################
            use_fullframe_layer = True
#############################################################
            if self.group_norm is not None:
            
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)  # (bf) x d(hw) x c
            self.q = query
            if self.inject_q is not None:
                # print(f"--- Inject q is {self.inject_q} ---") # NO Inject q
                query = self.inject_q
    
            dim = query.shape[-1]
            query_old = query.clone()

            # All frames
            #init query (bz*t, hxw, dim)
            # print(f"--- Query shape is {self.q.shape} ---")
            query = rearrange(query, "(b f) d c -> b (f d) c", f=clip_length)
            # print(f"--- Rearrange Query shape is {query.shape} ---")
            query = reshape_heads_to_batch_dim(query)  #(bz*heads, txhxw, dim//heads  [16, 61440, 40]
            # print(f"--- reshape_heads_to_batch Query shape is {query.shape} ---")
            if self.added_kv_proj_dim is not None:
                raise NotImplementedError
    
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

                
            key = self.to_k(encoder_hidden_states)
            self.k = key
            if self.inject_k is not None:
                key = self.inject_k
            key_old = key.clone()
            value = self.to_v(encoder_hidden_states)

            # print(f"encoder hidden state : {encoder_hidden_states.shape}") # [30, 4096, 320]
            value_old = value.clone()
            # print(f"value old : {value_old.shape}") # [30, 4096, 320]

            # if flow_only and not use_fullframe_layer:
            if not use_fullframe_layer:
                # print("323232323232323232323232323")
                encoder_hidden_states = rearrange(encoder_hidden_states, "(b f) d c -> b (f d) c", f=clip_length)
                hidden_states = encoder_hidden_states

            # if use_fullframe_layer or (not flow_only):
            if use_fullframe_layer:
                if inter_frame:
                    # print("full is using.")
                    # print("--- Inter frame is True ---")
                    key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
                    value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
                    key = rearrange(key, "b f d c -> b (f d) c",)
                    value = rearrange(value, "b f d c -> b (f d) c")
                else:
                    # All frames
                    # print("--- All frame is True ---")
                    key = rearrange(key, "(b f) d c -> b (f d) c", f=clip_length)
                    value = rearrange(value, "(b f) d c -> b (f d) c", f=clip_length)
    
                key = reshape_heads_to_batch_dim(key)
                value = reshape_heads_to_batch_dim(value)
                
                
                if attention_mask is not None:
                    if attention_mask.shape[-1] != query.shape[1]:
                        target_length = query.shape[1]
                        attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                        attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
                        print(f"--- attention mask shape is {attention_mask.shape}")  # no attention mask
                # else:
                #     print("--- Attention mask is none. ---")
        
                #print("query.shape[0]",query.shape[0])  # 16
                self._slice_size = 1   ### 8
                sequence_length_full_frame = query.shape[1]
    
                # attention, what we cannot get enough of
                if self._use_memory_efficient_attention_xformers and query.shape[-2] > clip_length*(32 ** 2):
                    print("--- fullframe attention ---")
                    ############### time_causal = True -> modified full frame attention ###############
                    hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask, time_causal , h, w, clip_length)
    
                    # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                    hidden_states = hidden_states.to(query.dtype)
                    encoder_hidden_states = hidden_states
                    # print(f"after full frame attn : {encoder_hidden_states.shape}")
                else:
                    if time_causal :
                        Q = sequence_length_full_frame  # = query.shape[-2]
                        # 幀級因果遮罩，允許同幀互看（若要嚴格只能看過去幀，把 include_same_frame=False）
                        attention_mask = build_frame_causal_mask(
                            sequence_length=Q,
                            clip_length=clip_length,
                            device=query.device,
                            dtype=query.dtype,
                            include_same_frame=True,
                        )

    
                        hidden_states = _sliced_attention(query, key, value, sequence_length_full_frame, dim, attention_mask, time_causal, h, w, clip_length)
                        encoder_hidden_states = hidden_states

        
### semantic flow attention
            # if controller.__class__.__name__ == "ST_Layout_Attn_ControlEdit" and not (h == 64 and w == 64):
            #     print("flow samentic attention==============")
            #     print(f"--- [h,w] : {[h,w]}, {kwargs['flatten_res']}")
            #     if [h, w] in kwargs['flatten_res']:
            #         print("--- start flow guided attention ---")
            
            #         # -------- Step 1: 前置準備 --------
            #         # (a) 把 encoder_hidden_states 攤回 (B*F, N, C) 供 flow 函式使用
            #         encoder_hidden_states_ff = rearrange(encoder_hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
            #         if self.group_norm is not None:
            #             encoder_hidden_states_ff = self.group_norm(encoder_hidden_states_ff.transpose(1, 2)).transpose(1, 2)
            
            #         # (b) old_qk / flow_only 的 q/k/v 選擇
            #         # if kwargs.get("old_qk", 1) == 1:
            #         #     query_old_ff = query_old
            #         #     key_old_ff   = key_old
            #         # else:
            #         query_old_ff = encoder_hidden_states_ff
            #         key_old_ff   = encoder_hidden_states_ff

            #         # if not flow_only or use_fullframe_layer:
            #         if use_fullframe_layer:
            #             # print("ababababababababababababababababababababa")
            #             value_old_ff = encoder_hidden_states_ff
            #         else:
            #             value_old_ff = value_old  # 直接沿用
            
            #         # (c) 正規化 traj/mask -> [F,N,L,*]，並裁到 anchor+過去 clip_length-1
            #         sequence_length = h * w  # N
            #         traj_in = kwargs["traj"]
            #         mask_in = kwargs["mask"]
            #         traj_ff, mask_ff = normalize_traj_and_mask(traj_in, mask_in, F_clip=clip_length, N=sequence_length)
            #         if traj_ff.size(2) >= clip_length:
            #             traj_ff = torch.cat([traj_ff[:, :, 0:1, :], traj_ff[:, :, -clip_length+1:, :]], dim=2)  # [F,N,L,3]
            #             mask_ff = torch.cat([mask_ff[:, :, 0:1],     mask_ff[:, :, -clip_length+1:]],     dim=2)  # [F,N,L]
            
            #         # (d) 準備 _key/_value 供沿軌跡 gather：[(B*F),N,D] -> [B,F,H,W,D]
            #         Bsmall = batch_size // clip_length
            #         _key_ff   = rearrange(key_old_ff,   '(b f) (hh ww) d -> b f hh ww d', b=Bsmall, f=clip_length, hh=h, ww=w)
            #         # value 用「若 flow_only 則 value_old；否則用 encoder_hidden_states_ff」的分支
            #         # value_for_gather = encoder_hidden_states_ff if not flow_only else value_old_ff
              
            #         _value_ff = rearrange(value_old_ff, '(b f) (hh ww) d -> b f hh ww d', b=Bsmall, f=clip_length, hh=h, ww=w)
            
            #         # -------- Step 2: 呼叫新 attention --------
            #         # low efficiency
            #         if use_fullframe_layer:
            #             flow_only=False
            #         else:
            #             flow_only=True
                        
            #         hidden_states_ff =  flow_semantic_traj_attention(
            #             query_old=query_old_ff,
            #             key_old=key_old_ff,
            #             # value_old=value_old_ff,
            #             encoder_hidden_states=encoder_hidden_states_ff,  # (B*F, N, C)
            #             group_norm=self.group_norm,
            #             traj=traj_ff,                  # [F, N, L, 3]
            #             mask=mask_ff,                  # [F, N, L]
            #             time_causal=time_causal,
            #             _key=_key_ff, _value=_value_ff,   # [B, F, H, W, D]
            #             h=h, w=w,
            #             clip_length=clip_length,
            #             heads=self.heads,
            #             controller=controller,
            #             sem_chunk=getattr(self, "sem_chunk", 64),
            #             use_sem_aug=getattr(self, "use_sem_aug", True),
            #             flow_only=flow_only,
            #             old_qk=kwargs.get("old_qk", 1),
            #         )
    
            #         # -------- Step 3: to_out 投影（與你原本一致） --------
            #         hidden_states_ff = self.to_out[0](hidden_states_ff)
            #         hidden_states_ff = self.to_out[1](hidden_states_ff)
            
            #         # -------- Step 4: 攤回 (B*F, N, C) 並 return --------
            #         hidden_states_ff = rearrange(hidden_states_ff, "b (f d) c -> (b f) d c", f=clip_length)
            #         return hidden_states_ff

            # else:
            ### original flow attention    
            # print(f"--- [h,w] : {[h,w]}, {kwargs['flatten_res']}")
            if [h,w] in kwargs['flatten_res']:
                # print("--- start flow guided attention ---")
                # if not flow_only:
                encoder_hidden_states = rearrange(encoder_hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
                if self.group_norm is not None:
                    encoder_hidden_states = self.group_norm(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
    
                if kwargs["old_qk"] == 1:
                    # print("--- old query and key for attention ---")
                    query = query_old
                    key = key_old
                    # print(f"flow query : {query.shape}") # [30, 4096, 320]
                    # print(f"flow key : {key.shape}") # [30, 4096, 320]
                # else:
                #     # print("--- hiden state for attention ---")
                #     query = encoder_hidden_states
                #     key = encoder_hidden_states
                # value = hidden_states
                # if not flow_only or use_fullframe_layer:
                if use_fullframe_layer:
                    value = encoder_hidden_states
                    # print(f"flow value : {value.shape}") # [30, 4096, 320]
                else:
                    # value_old = rearrange(value_old, "b (f d) c -> (b f) d c", f=clip_length)
                    value = value_old
                    # print(f"flow old : {value.shape}")
                
                traj = kwargs["traj"]
                traj = rearrange(traj, '(f n) l d -> f n l d', f=clip_length, n=sequence_length)
                
                mask = rearrange(kwargs["mask"], '(f n) l -> f n l', f=clip_length, n=sequence_length)
                mask = torch.cat([mask[:, :, 0].unsqueeze(-1), mask[:, :, -clip_length+1:]], dim=-1)
                # print(f"--- traj shape is {traj.shape} ---")  # traj shape is torch.Size([15, 4096, 39, 3])
                # print(f"--- mask shape is {mask.shape} ---")  # mask shape is torch.Size([15, 4096, 15])
                #print('traj',traj.shape)
                #print('mask',mask.shape)
    
                traj_key_sequence_inds = torch.cat([traj[:, :, 0, :].unsqueeze(-2), traj[:, :, -clip_length+1:, :]], dim=-2)
                t_inds = traj_key_sequence_inds[:, :, :, 0]
                x_inds = traj_key_sequence_inds[:, :, :, 1]
                y_inds = traj_key_sequence_inds[:, :, :, 2]
    
                ##### attention modification for previous frames #################################
                anchor = t_inds[:, :, 0].unsqueeze(-1).expand_as(t_inds)
    
                if time_causal:
                    traj_mask = t_inds <= anchor   
                else:
                    traj_mask = torch.ones_like(t_inds, dtype=torch.bool)
                t_inds = torch.where(traj_mask, t_inds, torch.zeros_like(t_inds))
                x_inds = torch.where(traj_mask, x_inds, torch.zeros_like(x_inds))
                y_inds = torch.where(traj_mask, y_inds, torch.zeros_like(y_inds))
                #############################################################
    

                query_tempo = query.unsqueeze(-2)   
                # print(f"--- query tempo shape: {query_tempo.shape} ---") # (2*15, 4096, 1, 320)
                _key = rearrange(key, '(b f) (h w) d -> b f h w d', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
                _value = rearrange(value, '(b f) (h w) d -> b f h w d', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
                # print(f"--- _key shape: {_key.shape} ---") # [2, 15, 64, 64, 320]
                # print(f"--- _value shape: {_value.shape} ---") # [2, 15, 64, 64, 320]
                key_tempo = _key[:, t_inds, x_inds, y_inds] #  [2, 15, 4096, 15, 320])
                value_tempo = _value[:, t_inds, x_inds, y_inds] # [2, 15, 4096, 15, 320])
                # print(f"--- key tempo shape: {key_tempo.shape} ---")
                # print(f"--- value tempo shape: {value_tempo.shape} ---")
                key_tempo = rearrange(key_tempo, 'b f n l d -> (b f) n l d') # [30, 4096, 15, 320]
                value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d') # [30, 4096, 15, 320]
                # print(f"--- key tempo shape: {key_tempo.shape} ---")
                # print(f"--- value tempo shape: {value_tempo.shape} ---")    
          
                ##### attention modification#################################
                seq_mask = mask
                # print(f"--- original mask shape : {seq_mask.shape}")
                # print(f"--- traj_mask shape : {traj_mask.shape}")
                keep_mask = seq_mask & traj_mask
                mask = rearrange(torch.stack([keep_mask, keep_mask]),  'b f n l -> (b f) n l')
    
                #############################################################
                # mask = rearrange(torch.stack([mask, mask]),  'b f n l -> (b f) n l')
                mask = mask[:,None].repeat(1, self.heads, 1, 1).unsqueeze(-2)
    
                
                attn_bias = torch.zeros_like(mask, dtype=key_tempo.dtype) # regular zeros_like
                attn_bias[~mask] = -torch.inf  
    
                # print('attn_bias',attn_bias.shape)  (30, H, 1, 4096, 15)
                # print('query_tempo',query_tempo.shape) # query_tempo torch.Size([30, 4096, 1, 320])
                # print('key_tempo',key_tempo.shape)     # key_tempo torch.Size([30, 4096, 15, 320])
                # print('value_tempo',value_tempo.shape) # value_tempo torch.Size([30, 4096, 15, 320])
                # flow attention
                query_tempo = reshape_heads_to_batch_dim3(query_tempo) # query_tempo torch.Size([30, 8, 4096, 1, 40])
                key_tempo = reshape_heads_to_batch_dim3(key_tempo)     # key_tempo torch.Size([30, 8, 4096, 15, 40])
                value_tempo = reshape_heads_to_batch_dim3(value_tempo) # value_tempo torch.Size([30, 8, 4096, 15, 40])
                # print("-------------------------------------------")
                # print('query_tempo',query_tempo.shape)
                # print('key_tempo',key_tempo.shape)
                # print('value_tempo',value_tempo.shape)
                
                attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(query_tempo.size(-1)) + attn_bias
                attn_matrix2 = F.softmax(attn_matrix2, dim=-1)
                out = (attn_matrix2@value_tempo).squeeze(-2)
    
                hidden_states = rearrange(out,'(b f) k (h w) d -> b (f h w) (k d)', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
    
            # linear proj
            hidden_states = self.to_out[0](hidden_states)
    
            # dropout
            hidden_states = self.to_out[1](hidden_states)
    
            # All frames
            hidden_states = rearrange(hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
            return hidden_states
                # # All frames




        if attention_type == 'CrossAttention':
            print("attention type: crossattention")
            # return mod_forward
            return forward
        elif attention_type == "SparseCausalAttention":
            print("attention type: sparsecausalattention")
            #return mod_forward
            return spatial_temporal_forward
        elif attention_type == "FullyFrameAttention":
            print("attention type: fullyframeattention")
            #return mod_forward
            return fully_frame_forward    

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()
    
    def register_recr(net_, count, place_in_unet):
        if net_[1].__class__.__name__ == 'CrossAttention' \
            or net_[1].__class__.__name__ == 'FullyFrameAttention' \
            or net_[1].__class__.__name__ == 'SparseCausalAttention' :
            net_[1].forward = attention_controlled_forward(net_[1], place_in_unet, attention_type = net_[1].__class__.__name__)
            return count + 1
        elif hasattr(net_[1], 'children'):
            for net in net_[1].named_children():
                if net[0] !='attn_temporal':

                    count = register_recr(net, count, place_in_unet)

        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    print(f"===============SubNet================= ")
    print(sub_nets)
    print("=======================================")
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net, 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net, 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net, 0, "mid")
    #print(f"Number of attention layer registered {cross_att_count}")
    controller.num_att_layers = cross_att_count



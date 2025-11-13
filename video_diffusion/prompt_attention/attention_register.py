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
import torch
import xformers
from xformers.ops import fmha

def register_attention_control(model, controller, text_cond, clip_length, height, width, ddim_inversion, id_masks_by_res=None, part_masks_by_res=None, dirty_A=None, dirty_B=None, clean_A=None, clean_B=None):
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

        def build_hybrid_mask(sequence_length, clip_length, device, dtype,
                              s_clean, b_clean):
            """
            回傳形狀為 (1, Q, K) 的 additive mask，允許=0，禁止=-inf
            - 對於 Q 來自「乾淨交集幀」：full-frame
            - 對於 Q 來自「非乾淨幀」：只能看到自己以及以前（時間因果；同幀僅 self）
            參數：
              sequence_length = t * hw
              clip_length = t (時間長度，幀數)
              s_clean, b_clean：兩個物體各自的乾淨幀索引（以幀為單位）
            """
            t = clip_length
            assert sequence_length % t == 0, "sequence_length 必須能被 clip_length 整除"
            hw = sequence_length // t
            n = sequence_length
        
            # 幀索引：每個 token 對應其所在幀
            # frame_ids: [0,0,...(hw 次), 1,1,..., 2,2,..., ..., t-1 重複 hw 次]
            frame_ids = torch.arange(t, device=device).repeat_interleave(hw)
            q_frame = frame_ids.view(n, 1).expand(n, n)   # (Q,K)
            k_frame = frame_ids.view(1, n).expand(n, n)   # (Q,K)
        
            # token 層級的單位矩陣：用來允許「同幀同 token 的 self」
            eye = torch.eye(n, dtype=torch.bool, device=device)
        
            # 「非乾淨幀」的允許規則：k_frame < q_frame（過去幀）或同一 token 自己
            allowed_dirty = (k_frame < q_frame) | eye
        
            # 初始全不允許
            allowed = torch.zeros((n, n), dtype=torch.bool, device=device)
        
            # 乾淨交集幀（以幀為單位）
            clean_frames = sorted(set(s_clean) & set(b_clean))
            clean_frames = torch.tensor(clean_frames, device=device, dtype=torch.long) if len(clean_frames) > 0 \
                           else torch.tensor([], device=device, dtype=torch.long)
        
            # 將幀索引映射到 token 索引範圍
            if clean_frames.numel() > 0:
                # 這些 Q-row（來自乾淨交集幀）採 full-frame（允許全部 K）
                # 每個幀對應的 Q 索引範圍是 [f*hw, (f+1)*hw)
                clean_q_rows = torch.cat([torch.arange(f*hw, (f+1)*hw, device=device) for f in clean_frames])
                allowed[clean_q_rows, :] = True
        
            # 其餘 Q-row（非乾淨幀）採因果遮罩：只能自己及以前
            dirty_q_rows_mask = torch.ones(n, dtype=torch.bool, device=device)
            if clean_frames.numel() > 0:
                dirty_q_rows_mask[clean_q_rows] = False
            dirty_q_rows = dirty_q_rows_mask.nonzero(as_tuple=False).squeeze(-1)
            if dirty_q_rows.numel() > 0:
                allowed[dirty_q_rows, :] = allowed_dirty[dirty_q_rows, :]
        
            # 轉成 additive mask：允許=0，禁止=-inf（用 float32 以避免精度問題）
            finfo = torch.finfo(torch.float32 if dtype == torch.float16 else torch.float32)
            additive = torch.where(
                allowed,
                torch.zeros(1, device=device, dtype=torch.float32),
                torch.full((1,), finfo.min, device=device, dtype=torch.float32)
            ).view(1, n, n)
        
            return additive
        def _memory_efficient_attention_xformers(query, key, value, attention_mask, time_causal=False, s_clean=None, b_clean=None, clip_length=None):
            # TODO attention_mask
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            ##################################################
            BxH, Q, Dq = query.shape
            _,   K, Dk = key.shape
            assert Dq == Dk, f"Q/K dim mismatch: {Dq} vs {Dk}"

            attn_bias = None
            if time_causal:
                from xformers.ops import fmha
                causal_bias = fmha.attn_bias.LowerTriangularMask()
                attn_bias = causal_bias
            ##################################################
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attn_bias)
            hidden_states = reshape_batch_dim_to_heads(hidden_states)
            return hidden_states 

    
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            # hidden_states: torch.Size([16, 4096, 320])
            # encoder_hidden_states: torch.Size([16, 77, 768])
            # print("========================= Cross Attentoin ===============================")
            is_cross = encoder_hidden_states is not None
            
            #encoder_hidden_states = encoder_hidden_states

            text_cond_frames = text_cond.repeat_interleave(clip_length, 0)     # wrong implementation text_cond.repeat(clip_length,1,1)

            ######for debug######
            # text_cond_repeat_interleave = text_cond.repeat_interleave(clip_length, 0)
            # print("after repeat interleave", text_cond_repeat_interleave.shape, text_cond_repeat_interleave.view(-1)[:20])
            # text_cond_repeat = text_cond.repeat(clip_length,1,1)
            # print("First 20 elements after repeat:", text_cond_repeat.shape, text_cond_repeat.view(-1)[:20])
            ######for debug######

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
                # for large attention map of 64X64, use xformers to save memory
                # print("xformers")
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
            # print("increase cross attention.")
            # CA_ALPHA = 2.0  # 0.1~0.3 常見；想再弱就更小
            # hidden_states = hidden_states * CA_ALPHA
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


##Great!!!!
        def _sliced_attention(query, key, value, sequence_length, dim, attention_mask, time_causal,
                              height=None, width=None, clip_length=None):
            enable_record = False
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
        
            # # 你要的設定：參考幀與交錯起點
            # REF_FRAMES = list(range(0, 3))  # 0..5 幀
            # CROSS_START = 3  # 從第 6 幀開始視為交錯段
            # record_after_softmax = True  # 統計 old/new 時是否記機率（維持你的原習慣）
            def _sanitize(frames):
                if frames is None: 
                    return []
                # 去重、排序、且限制在 [0, F-1]
                fs = sorted(set(int(x) for x in frames if 0 <= int(x) < F))
                return fs
            
            S_clean = _sanitize(clean_A)  # 例如 [0,1,2,3,4,5]
            B_clean = _sanitize(clean_B)  # 例如 [0,1,2,3,4,5]
            S_dirty = _sanitize(dirty_A)  # 例如 [6..14]
            B_dirty = _sanitize(dirty_B)  # 例如 [5..14]   
    
            
            # ---- 新的觸發條件：各做各的 ----
            DO_XFORM_S = len(S_dirty) > 0   # A(S) 有 dirty 就處理 A
            DO_XFORM_B = len(B_dirty) > 0   # B 有 dirty 才處理 B
            DO_XFORM   = DO_XFORM_S or DO_XFORM_B
            print(f"DO_XFORM_S={DO_XFORM_S}, DO_XFORM_B={DO_XFORM_B}")

            print(f"s_clean: {S_clean}")
            print(f"B_clean: {B_clean}")
            print(f"S_dirty: {S_dirty}")
            print(f"B_dirty: {B_dirty}")
            ###交換與縮放強度
            EXCHANGE_STRENGTH = 2.0             # 均值交換強度（維持你原本邏輯）
            SHRINK_FACTOR    = 0.5  # <<<< [ADD] 縮小跨類注意力（StoB / BtoS）之倍率（0~1）
            SHRINK_ON_TAIL   = True       # <<<< [ADD] 僅在發現 "跨類 > 同類" 時才縮
            # <<< 新增：同類放大 >>>
            BOOST_FACTOR     = 1.25    # 同類注意力（StoS / BtoB）放大倍率（>1）
            BOOST_ON_TAIL    = False     # 建議跟 SHRINK_ON_TAIL 一樣條件觸發       
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
                # for r in REF_FRAMES:
                    # S_cols_ref_union |= _frame_col_mask("S", r)
                    # B_cols_ref_union |= _frame_col_mask("B", r)
                for r in S_clean:
                    S_cols_ref_union |= _frame_col_mask("S", r)
                for r in B_clean:
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
        
            def _exchange_by_mean_shift(attn_logits, row_mask, col_mask_a, col_mask_b, strength=1.0, eps=1e-8):
                slice_len, Q, K = attn_logits.shape
                Ma = _mk3(row_mask, col_mask_a, slice_len, Q, K)  # bool
                Mb = _mk3(row_mask, col_mask_b, slice_len, Q, K)  # bool
            
                # 用 logits 的 dtype 計算
                Ma_f = Ma.to(dtype=attn_logits.dtype)
                Mb_f = Mb.to(dtype=attn_logits.dtype)
            
                sum_a = (attn_logits * Ma_f).sum(dim=-1)
                cnt_a = Ma_f.sum(dim=-1).clamp_min(eps)
                mu_a = sum_a / cnt_a
            
                sum_b = (attn_logits * Mb_f).sum(dim=-1)
                cnt_b = Mb_f.sum(dim=-1).clamp_min(eps)
                mu_b = sum_b / cnt_b
            
                row_mask_Q = row_mask.unsqueeze(0).expand(slice_len, -1)  # bool
                delta = (mu_b - mu_a) * strength
                delta = torch.where(row_mask_Q, delta, torch.zeros_like(delta))
                delta = delta.to(attn_logits.dtype)  # 關鍵：轉成 half
            
                # 寫回時兩邊 dtype 一致
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
            def _boost_block(attn_logits: torch.Tensor, row_mask: torch.Tensor, col_mask: torch.Tensor, factor: float):
                """
                僅放大 row_mask × col_mask 子塊： attn[row,col] *= factor
                - 建議 factor > 1.0
                - 與 _shrink_block 相反，但仍在 softmax 之前於 logits 空間就地乘上係數
                """
                if factor <= 1.0:
                    return
                slice_len, Q, K = attn_logits.shape
                M = _mk3(row_mask, col_mask, slice_len, Q, K)
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
        
            def _compute_baseline_sb_over_ss(attn_logits: torch.Tensor):
                if S_cols_ref_union is None:
                    return None
                vals = []
                eps = 1e-6
                for t in S_clean:  # 只看 S 的 clean 幀
                    means_j, _, _ = _pair_means_for_t_perhead(attn_logits, t)
                    if means_j is None: 
                        continue
                    ss = means_j[:, 0]; sb = means_j[:, 1]
                    vals.append( sb / (ss.abs() + eps) )
                if not vals:
                    return None
                return torch.stack(vals, dim=0).nanmean(dim=0)  # [slice_len]
            
            def _compute_baseline_bs_over_bb(attn_logits: torch.Tensor):
                if B_cols_ref_union is None:
                    return None
                vals = []
                eps = 1e-6
                for t in B_clean:  # 只看 B 的 clean 幀
                    means_j, _, _ = _pair_means_for_t_perhead(attn_logits, t)
                    if means_j is None: 
                        continue
                    bb = means_j[:, 3]; bs = means_j[:, 2]
                    vals.append( bs / (bb.abs() + eps) )
                if not vals:
                    return None
                return torch.stack(vals, dim=0).nanmean(dim=0)  # [slice_len]      
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
        
                attn_slice_old = attn_slice.clone().detach()  # 交換前快照 (logits)
                if DO_XFORM:
                    print("===do transform===")
                    # baseline（逐 local_j/head）
                    baseline_sb_ss = _compute_baseline_sb_over_ss(attn_slice_old)  # for S rows
                    baseline_bs_bb = _compute_baseline_bs_over_bb(attn_slice_old)  # for B rows
                    
                    base_ratio_min = 1.00
                    delta_thr = 0.0
                    eps = 1e-6
                    max_heads_to_fix = max(1, self.heads // 2)
                    
                    if (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
                        if DO_XFORM_S:
                            # ---------- S 行：只處理 dirty_A ----------
                            for t in S_dirty:
                                means_j, S_row_t, B_row_t = _pair_means_for_t_perhead(attn_slice, t)
                                if means_j is None:
                                    continue
                                ss = means_j[:, 0]; sb = means_j[:, 1]
                                now_sb_ss = sb / (ss.abs() + eps)
                        
                                if baseline_sb_ss is not None:
                                    thr_sb_ss = baseline_sb_ss * (1.0 + delta_thr)
                                else:
                                    thr_sb_ss = torch.full_like(now_sb_ss, base_ratio_min)
                        
                                viol_S = (now_sb_ss > torch.maximum(thr_sb_ss, torch.full_like(thr_sb_ss, base_ratio_min)))
                                score_S = (now_sb_ss - thr_sb_ss).nan_to_num(0.0) if baseline_sb_ss is not None else (now_sb_ss - base_ratio_min).nan_to_num(0.0)
                        
                                idx_S = torch.nonzero(viol_S, as_tuple=False).flatten()
                                if idx_S.numel() > 0:
                                    topk = min(max_heads_to_fix, idx_S.numel())
                                    topS = idx_S[torch.topk(score_S[idx_S], k=topk, largest=True).indices]
                                    for j in topS:
                                        _exchange_by_mean_shift(
                                            attn_slice[j:j+1],
                                            row_mask=S_row_t,
                                            col_mask_a=S_cols_ref_union,  # 正確（S_to_S）
                                            col_mask_b=B_cols_ref_union,  # 錯誤（S_to_B）
                                            strength=EXCHANGE_STRENGTH,
                                            eps=eps,
                                        )
                        
                                # 交換後如已成功，視情況縮 StoB
                                if SHRINK_ON_TAIL and idx_S.numel() > 0:
                                    print("shrink")
                                    means_after, _, _ = _pair_means_for_t_perhead(attn_slice, t)
                                    if means_after is not None:
                                        ss2 = means_after[:, 0]; sb2 = means_after[:, 1]
                                        success_S = (ss2 > sb2)
                                        for j in torch.nonzero(success_S, as_tuple=False).flatten():
                                            _shrink_block(attn_slice[j:j+1], S_row_t, B_cols_ref_union, factor=SHRINK_FACTOR)
                                            if BOOST_ON_TAIL:
                                                _boost_block(attn_slice[j:j+1], S_row_t, S_cols_ref_union, factor=BOOST_FACTOR)                    
                        # ---------- B 行：只處理 dirty_B ----------
                        if DO_XFORM_B:
                            for t in B_dirty:
                                means_j, S_row_t, B_row_t = _pair_means_for_t_perhead(attn_slice, t)
                                if means_j is None:
                                    continue
                                bb = means_j[:, 3]; bs = means_j[:, 2]
                                now_bs_bb = bs / (bb.abs() + eps)
                        
                                if baseline_bs_bb is not None:
                                    thr_bs_bb = baseline_bs_bb * (1.0 + delta_thr)
                                else:
                                    thr_bs_bb = torch.full_like(now_bs_bb, base_ratio_min)
                        
                                viol_B = (now_bs_bb > torch.maximum(thr_bs_bb, torch.full_like(thr_bs_bb, base_ratio_min)))
                                score_B = (now_bs_bb - thr_bs_bb).nan_to_num(0.0) if baseline_bs_bb is not None else (now_bs_bb - base_ratio_min).nan_to_num(0.0)
                        
                                idx_B = torch.nonzero(viol_B, as_tuple=False).flatten()
                                if idx_B.numel() > 0:
                                    topk = min(max_heads_to_fix, idx_B.numel())
                                    topB = idx_B[torch.topk(score_B[idx_B], k=topk, largest=True).indices]
                                    for j in topB:
                                        _exchange_by_mean_shift(
                                            attn_slice[j:j+1],
                                            row_mask=B_row_t,
                                            col_mask_a=B_cols_ref_union,  # 正確（B_to_B）
                                            col_mask_b=S_cols_ref_union,  # 錯誤（B_to_S）
                                            strength=EXCHANGE_STRENGTH,
                                            eps=eps,
                                        )
                        
                                # 交換後如已成功，視情況縮 BtoS
                                if SHRINK_ON_TAIL and idx_B.numel() > 0:
                                    print("shrink")
                                    means_after, _, _ = _pair_means_for_t_perhead(attn_slice, t)
                                    if means_after is not None:
                                        bb2 = means_after[:, 3]; bs2 = means_after[:, 2]
                                        success_B = (bb2 > bs2)
                                        for j in torch.nonzero(success_B, as_tuple=False).flatten():
                                            _shrink_block(attn_slice[j:j+1], B_row_t, S_cols_ref_union, factor=SHRINK_FACTOR)
                                            if BOOST_ON_TAIL:
                                                _boost_block(attn_slice[j:j+1], B_row_t, B_cols_ref_union, factor=BOOST_FACTOR)
        
                attn_slice_exch = attn_slice.clone().detach()  # 交換/縮放後、controller 前的快照 (exch, logits)
        
                controller_applied = False
                if i < self.heads and not ddim_inversion:
                    print("in controller")
                    attention_out = controller(attn_slice.unsqueeze(1), is_cross, place_in_unet, height, width, clip_length)
                    attn_slice = attention_out.squeeze(1)
                    controller_applied = True
        
        
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
            return hidden_states
##########################################################################################################





### original

        def fully_frame_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, clip_length=None, inter_frame=False, flow_only=True, time_causal=True, **kwargs):
            # print(" ====== attn1 is displayed by attention register (attention_register.py line 377) ======")
            # print(encoder_hidden_states) # None
            print(f"time causal: {time_causal}")
            batch_size, sequence_length, _ = hidden_states.shape
            # print("hidden_states.shape",hidden_states.shape)
            # print("sequence_length",sequence_length)
            # print("======================== Full Frame forward ========================")
            encoder_hidden_states = encoder_hidden_states
            h = kwargs['height']
            w = kwargs['width']

            use_fullframe_layer = True

            if self.group_norm is not None:
                # print("group norm") # no group norm
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
                    hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask, time_causal, clean_A, clean_B, clip_length)
                    # hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask)
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
    
                
                # for i in range(14):
                #     print(f"--- tinds : {t_inds.shape}, {t_inds[i, 2886]}") # (15, 4096, 15)
                #     print(f"--- xinds : {x_inds.shape}, {x_inds[i, 2886]}")
                #     print(f"--- yinds : {y_inds.shape}, {y_inds[i, 2886]}")
                #     print("-" * 25)
                # for j in range(2000, 2020, 1):
                #     print(f"--- tinds : {t_inds.shape}, {t_inds[7, j]}") # (15, 4096, 15)
                #     print(f"--- xinds : {x_inds.shape}, {x_inds[7, j]}")
                #     print(f"--- yinds : {y_inds.shape}, {y_inds[7, j]}")
                #     print("+" * 25)
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

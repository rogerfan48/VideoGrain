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

        # def _memory_efficient_attention_xformers(
        #     query, key, value, attention_mask=None, time_causal=False,
        #     s_clean=None, b_clean=None, clip_length=None
        # ):
        #     """
        #     query/key/value: (B*H, Q, D)
        #     規則：
        #       - 乾淨交集幀 (s_clean ∩ b_clean) -> full-frame
        #       - 其他幀 -> 只能看自己與過去：過去幀全可見；同幀僅 self
        #     不再建立全域 (1, Q, K) bias，改為拆批：
        #       - 乾淨交集幀：一次性跑 full attention
        #       - 非乾淨幀：逐幀拼接 K=[past ; same_frame] 與小型對角遮罩
        #     """
        #     query = query.contiguous()
        #     key   = key.contiguous()
        #     value = value.contiguous()
        
        #     BxH, Q, Dq = query.shape
        #     _,   K, Dk = key.shape
        #     assert Dq == Dk, f"Q/K dim mismatch: {Dq} vs {Dk}"
        #     assert Q == K, "目前假設自注意力 Q==K"
        #     assert clip_length is not None, "需要 clip_length=t"
        #     device = query.device
        
        #     # 每幀 token 數
        #     t = clip_length
        #     assert Q % t == 0, "sequence_length 必須能被 clip_length 整除"
        #     hw = Q // t
        
        #     # 幀到 token 的區段切片
        #     def fr_slice(f):
        #         start = f * hw
        #         end   = (f + 1) * hw
        #         return slice(start, end)
        
        #     # 乾淨交集幀 / 非乾淨幀
        #     s_clean = set([] if s_clean is None else s_clean)
        #     b_clean = set([] if b_clean is None else b_clean)
        #     clean_frames = sorted(s_clean & b_clean)
        #     all_frames   = set(range(t))
        #     dirty_frames = sorted(all_frames - set(clean_frames))
        
        #     # 收集輸出的容器
        #     out = torch.empty_like(query)
        
        #     # -------- 1) 乾淨交集幀：一次性 full attention（不需要任何 bias）--------
        #     if len(clean_frames) > 0:
        #         clean_rows = []
        #         for f in clean_frames:
        #             sl = fr_slice(f)
        #             clean_rows.append(torch.arange(sl.start, sl.stop, device=device))
        #         clean_rows = torch.cat(clean_rows, dim=0)  # (Nc,)
        
        #         q_clean = query[:, clean_rows, :]  # (B*H, Nc, D)
        #         # full attention：允許所有 K
        #         h_clean = xformers.ops.memory_efficient_attention(q_clean, key, value, attn_bias=None)
        #         out[:, clean_rows, :] = h_clean
        
        #     # -------- 2) 非乾淨幀：逐幀處理，只看過去 + 同幀 self --------
        #     # 對於幀 f：
        #     #   K_cat = [K_past(0..f*hw-1), K_curr(f*hw..(f+1)*hw-1)]
        #     #   對 K_curr 施加 (hw×hw) 的對角遮罩，只允許 self，其餘 -inf
        #     BxH = query.shape[0]
        #     q_dtype = query.dtype
        #     finfo = torch.finfo(q_dtype)
        
        #     for f in dirty_frames:
        #         q_rows = fr_slice(f)
        #         q_f = query[:, q_rows, :]
        
        #         past_len = f * hw
        #         K_past = key[:, :past_len, :]
        #         V_past = value[:, :past_len, :]
        #         K_curr = key[:, q_rows, :]
        #         V_curr = value[:, q_rows, :]
        
        #         K_cat = torch.cat([K_past, K_curr], dim=1)
        #         V_cat = torch.cat([V_past, V_curr], dim=1)
        
        #         # (hw, past_len+hw) additive bias, 預設 -inf
        #         bias_base = torch.full((hw, past_len + hw), finfo.min, device=device, dtype=q_dtype)
        
        #         # 過去幀：每個 i 只允許同一空間位置 i 的歷史（i, i+hw, i+2*hw, ...）
        #         if past_len > 0:
        #             cols = torch.arange(0, past_len, hw, device=device)            # (f,)
        #             row_idx = torch.arange(hw, device=device).unsqueeze(1)          # (hw,1)
        #             col_idx = cols.unsqueeze(0) + torch.arange(hw, device=device).unsqueeze(1)  # (hw,f)
        #             bias_base[row_idx, col_idx] = 0
        
        #         # 同幀：只允許 self（右半塊對角）
        #         right_diag = past_len + torch.arange(hw, device=device)
        #         bias_base[torch.arange(hw, device=device), right_diag] = 0
        
        #         # ❶ 批次維度展開
        #         attn_bias = bias_base.unsqueeze(0).expand(BxH, -1, -1)
        #         # 若仍報 shape 錯，改用 4D：attn_bias = attn_bias.unsqueeze(1)  # (BxH, 1, hw, Kc)
        
        #         h_f = xformers.ops.memory_efficient_attention(q_f, K_cat, V_cat, attn_bias=attn_bias)
        #         out[:, q_rows, :] = h_f

        
        #     # （選擇性）如果你想保留舊的嚴格時間因果選項
        #     if time_causal and len(clean_frames) == 0 and len(dirty_frames) == 0:
        #         # 沒傳入 clean/dirty，且要求 causal，回退到下三角
        #         causal_bias = fmha.attn_bias.LowerTriangularMask()
        #         out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=causal_bias)
        
        #     hidden_states = reshape_batch_dim_to_heads(out)
        #     return hidden_states


    
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
    
        # def _sliced_attention(query, key, value, sequence_length, dim, attention_mask, time_causal, height=None, width=None, clip_length=None):
        #     #query (bz*heads, t x h x w, org_dim//heads )
        #     # print("================== _sliced attention ====================")
        #     is_cross = False
        #     batch_size_attention = query.shape[0]   # bz * heads
        #     hidden_states = torch.zeros(
        #         (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        #     )

        #     slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]

        #     if ddim_inversion:
        #         per_frame_len = sequence_length//clip_length
        #         attention_store = torch.zeros((batch_size_attention, clip_length, per_frame_len, per_frame_len), device=query.device, dtype=query.dtype)

        #     for i in range(hidden_states.shape[0] // slice_size):
        #         start_idx = i * slice_size
        #         end_idx = (i + 1) * slice_size

        #         query_slice = query[start_idx:end_idx]
        #         key_slice = key[start_idx:end_idx]

        #         if self.upcast_attention:
        #             query_slice = query_slice.float()
        #             key_slice = key_slice.float()

        #         attn_slice = torch.baddbmm(
        #             torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
        #             query_slice,
        #             key_slice.transpose(-1, -2),
        #             beta=0,
        #             alpha=self.scale,
        #         )

        #         if attention_mask is not None:
        #             if time_causal:
        #                 attn_slice = attn_slice + attention_mask[:1]
        #             else:
        #                 attn_slice = attn_slice + attention_mask[start_idx:end_idx]

        #         if self.upcast_softmax:
        #             attn_slice = attn_slice.float()

        #         if i < self.heads:
        #             if not ddim_inversion:
        #                 attention_probs = controller((attn_slice.unsqueeze(1)),is_cross, place_in_unet)
        #                 attn_slice = attention_probs.squeeze(1)

        #         attn_slice = attn_slice.softmax(dim=-1)

        #         # cast back to the original dtype
        #         attn_slice = attn_slice.to(value.dtype)
        #         ## bz == 1, sliced head 
        #         if ddim_inversion:
        #             # attn_slice (1, thw, thw)
        #             bz, thw, thw = attn_slice.shape
        #             t = clip_length
        #             hw =  thw // t
        #             # 初始化 per_frame_attention
        #             # (1, t, hxw)

        #             per_frame_attention = torch.empty((bz, t, hw, hw), device=attn_slice.device)

        #             # # 循环提取每一帧的对角线注意力
        #             for idx in range(t):
        #                 start_idx_ = idx * hw
        #                 end_idx_ = (idx + 1) * hw
        #                 # per frame attention extraction
        #                 per_frame_attention[:, idx, :, :] = attn_slice[:, start_idx_:end_idx_, start_idx_:end_idx_]

        #                 # current_query_block = attn_slice[:, start_idx_:end_idx_, :] 
        #                 # aggregated_attention = current_query_block.view(bz, hw, t, hw).mean(dim=2)
        #                 # # print('aggregated_attention',aggregated_attention.shape)
        #                 # per_frame_attention[:, idx, :, :] = aggregated_attention

        #             per_frame_attention = rearrange(per_frame_attention, "b t h w -> (b t) h w")
        #             attention_store[start_idx:end_idx] = per_frame_attention
                
        #         attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

        #         hidden_states[start_idx:end_idx] = attn_slice
        #     if ddim_inversion:
        #         # attention store (bz*heads, t , h, w) h=res, w=res
        #         _ = controller(attention_store, is_cross, place_in_unet)

        #     # reshape hidden_states
        #     hidden_states = reshape_batch_dim_to_heads(hidden_states)
        #     return hidden_states

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
        
            ### 參考幀（0..5）之列遮罩聯集（Key 軸）
            # print("[DBG] id_masks_by_res keys:", list(id_masks_by_res.keys()))
            # for k, v in id_masks_by_res.items():
            #     print(f"  {k}: type={type(v)}")
            #     if isinstance(v, dict):
            #         print("    subkeys:", list(v.keys()))
            #         for subk, subv in v.items():
            #             if isinstance(subv, torch.Tensor):
            #                 print(f"      {subk}: shape={tuple(subv.shape)} dtype={subv.dtype}")
            #             else:
            #                 print(f"      {subk}: type={type(subv)}")
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
        
            # def _exchange_by_mean_shift(
            #     attn_logits: torch.Tensor,
            #     row_mask: torch.Tensor,
            #     col_mask_a: torch.Tensor,
            #     col_mask_b: torch.Tensor,
            #     strength: float = 1.0,
            #     eps: float = 1e-8,
            # ):
            #     """
            #     對 attn_logits 的兩個子區塊 A(row_mask,col_mask_a) 與 B(row_mask,col_mask_b) 進行「均值交換/靠攏」：
            #     A <- A + strength * (μ_B - μ_A)
            #     B <- B - strength * (μ_B - μ_A)
            #     """
            #     slice_len, Q, K = attn_logits.shape
            #     Ma = _mk3(row_mask, col_mask_a, slice_len, Q, K)
            #     Mb = _mk3(row_mask, col_mask_b, slice_len, Q, K)
            #     Ma_f = Ma.float()
            #     Mb_f = Mb.float()
        
            #     sum_a = (attn_logits * Ma_f).sum(dim=-1)  # [slice, Q]
            #     cnt_a = Ma_f.sum(dim=-1).clamp_min(eps)   # [slice, Q]
            #     mu_a = sum_a / cnt_a
        
            #     sum_b = (attn_logits * Mb_f).sum(dim=-1)  # [slice, Q]
            #     cnt_b = Mb_f.sum(dim=-1).clamp_min(eps)   # [slice, Q]
            #     mu_b = sum_b / cnt_b
        
            #     row_mask_Q = row_mask.unsqueeze(0).expand(slice_len, -1)  # [slice, Q]
            #     delta = (mu_b - mu_a) * strength
            #     delta = torch.where(row_mask_Q, delta, torch.zeros_like(delta))
        
            #     # 把 delta 寫回兩塊（逐元素）
            #     attn_logits[Ma] = (attn_logits[Ma] + delta.unsqueeze(-1).expand_as(attn_logits).masked_select(Ma)).reshape(-1)
            #     attn_logits[Mb] = (attn_logits[Mb] - delta.unsqueeze(-1).expand_as(attn_logits).masked_select(Mb)).reshape(-1)
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
        
            # baseline（避免天生跨類的 head 被誤判）
            # def _compute_baseline_ratios_perhead(attn_logits: torch.Tensor):
            #     if S_cols_ref_union is None:
            #         return None, None
            #     sb_over_ss = []
            #     bs_over_bb = []
            #     eps = 1e-6
            #     for t in REF_FRAMES:
            #         means_j, _, _ = _pair_means_for_t_perhead(attn_logits, t)
            #         if means_j is None:
            #             continue
            #         ss = means_j[:, 0]; sb = means_j[:, 1]; bs = means_j[:, 2]; bb = means_j[:, 3]
            #         sb_over_ss.append( (sb / (ss.abs()+eps)) )
            #         bs_over_bb.append( (bs / (bb.abs()+eps)) )
            #     if len(sb_over_ss) == 0:
            #         return None, None
            #     baseline_sb_ss = torch.stack(sb_over_ss, dim=0).nanmean(dim=0)  # [slice_len]
            #     baseline_bs_bb = torch.stack(bs_over_bb, dim=0).nanmean(dim=0)
            #     return baseline_sb_ss, baseline_bs_bb
            # 分別得到 S 與 B 的 baseline ratio
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
        
                # attn_slice_old = attn_slice.clone().detach()  # 交換前的快照 (old, logits)
        
                # # baseline（逐 local_j/head）
                # baseline_sb_ss, baseline_bs_bb = _compute_baseline_ratios_perhead(attn_slice_old)
        
                # # ------------------ [MOD] 交換條件：逐 head/逐 slice-row 判斷 ------------------
                # base_ratio_min = 1.00    # 最低相對倍率門檻
                # delta_thr = 0.0          # 比 baseline 至少高 25% 才觸發
                # eps = 1e-6
                # max_heads_to_fix = max(1, self.heads // 2)
                
                # if (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
                #     for t in range(CROSS_START, F):
                #         means_j, S_row_t, B_row_t = _pair_means_for_t_perhead(attn_slice, t)
                #         if means_j is None:
                #             continue
                #         ss = means_j[:, 0]; sb = means_j[:, 1]; bs = means_j[:, 2]; bb = means_j[:, 3]
                #         # print("[CHK] slice_len:", means_j.shape[0], " slice_size=", slice_size)
                #         now_sb_ss = sb / (ss.abs() + eps)
                #         now_bs_bb = bs / (bb.abs() + eps)
        
                #         thr_sb_ss = (baseline_sb_ss * (1.0 + delta_thr)) if baseline_sb_ss is not None else torch.full_like(now_sb_ss, base_ratio_min)
                #         thr_bs_bb = (baseline_bs_bb * (1.0 + delta_thr)) if baseline_bs_bb is not None else torch.full_like(now_bs_bb, base_ratio_min)
        
                #         viol_S = (now_sb_ss > torch.maximum(thr_sb_ss, torch.full_like(thr_sb_ss, base_ratio_min)))
                #         viol_B = (now_bs_bb > torch.maximum(thr_bs_bb, torch.full_like(thr_bs_bb, base_ratio_min)))
        
                #         score_S = (now_sb_ss - thr_sb_ss).nan_to_num(0.0) if baseline_sb_ss is not None else (now_sb_ss - base_ratio_min).nan_to_num(0.0)
                #         score_B = (now_bs_bb - thr_bs_bb).nan_to_num(0.0) if baseline_bs_bb is not None else (now_bs_bb - base_ratio_min).nan_to_num(0.0)
        
                #         idx_S = torch.nonzero(viol_S, as_tuple=False).flatten()
                #         idx_B = torch.nonzero(viol_B, as_tuple=False).flatten()
        
                #         # ===== 第一步：先做「均值交換」把注意力拉回同類 =====
                #         if idx_S.numel() > 0:
                #             topk = min(max_heads_to_fix, idx_S.numel())
                #             # print(f"max head to fix:{max_heads_to_fix}, idxs:{idx_S.numel()}, topk:{topk}")
                #             topS = idx_S[torch.topk(score_S[idx_S], k=topk, largest=True).indices]
                #             for j in topS:
                #                 _exchange_by_mean_shift(
                #                     attn_slice[j:j+1],
                #                     row_mask=S_row_t,
                #                     col_mask_a=S_cols_ref_union,  # 正確（StoS）
                #                     col_mask_b=B_cols_ref_union,  # 錯誤（StoB）
                #                     strength=EXCHANGE_STRENGTH,
                #                     eps=eps,
                #                 )
        
                #         if idx_B.numel() > 0:
                #             topk = min(max_heads_to_fix, idx_B.numel())
                #             # print(f"max head to fix:{max_heads_to_fix}, idxs:{idx_B.numel()}, topk:{topk}")
                #             topB = idx_B[torch.topk(score_B[idx_B], k=topk, largest=True).indices]
                #             for j in topB:
                #                 _exchange_by_mean_shift(
                #                     attn_slice[j:j+1],
                #                     row_mask=B_row_t,
                #                     col_mask_a=B_cols_ref_union,  # 正確（BtoB）
                #                     col_mask_b=S_cols_ref_union,  # 錯誤（BtoS）
                #                     strength=EXCHANGE_STRENGTH,
                #                     eps=eps,
                #                 )
        
                #        # ===== 第二步：若交換後仍「跨類 > 同類」，就縮小跨類權重（乘 SHRINK_FACTOR） =====
                #         # ===== 第二步 [CHANGE]：若「交換後已成功」（同類 > 跨類），才縮小跨類權重 =====
                #         if SHRINK_ON_TAIL and (idx_S.numel() > 0 or idx_B.numel() > 0):
                #             means_after, _, _ = _pair_means_for_t_perhead(attn_slice, t)
                #             if means_after is not None:
                #                 ss2 = means_after[:, 0]; sb2 = means_after[:, 1]
                #                 bs2 = means_after[:, 2]; bb2 = means_after[:, 3]
        
                #                 # 成功條件：StoS > StoB、BtoB > BtoS
                #                 success_S = (ss2 > sb2)  # S 行成功
                #                 success_B = (bb2 > bs2)  # B 行成功
        
                #                 # 只在「成功」的 head 上進一步縮小跨類注意力
                #                 idx_success_S = torch.nonzero(success_S, as_tuple=False).flatten()
                #                 idx_success_B = torch.nonzero(success_B, as_tuple=False).flatten()
        
                #                 for j in idx_success_S:
                #                     # 抑制 S row × (B cols) -> 縮 StoB
                #                     # print("shrink")
                #                     _shrink_block(attn_slice[j:j+1], S_row_t, B_cols_ref_union, factor=SHRINK_FACTOR)
        
                #                 for j in idx_success_B:
                #                     # 抑制 B row × (S cols) -> 縮 BtoS
                #                     # print("shrink")
                #                     _shrink_block(attn_slice[j:j+1], B_row_t, S_cols_ref_union, factor=SHRINK_FACTOR)
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
        
                # ------------------ 統計蒐集：old / exch 放在 t 迴圈內逐幀收 ------------------
                # if (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
                #     num_heads = self.heads
                #     for t in range(CROSS_START, F):
                #         try:
                #             S_row_t = _frame_row_mask("S", t)
                #             B_row_t = _frame_row_mask("B", t)
                #         except KeyError:
                #             continue
                #         pm = {
                #             "S_to_S": (S_row_t, S_cols_ref_union),
                #             "S_to_B": (S_row_t, B_cols_ref_union),
                #             "B_to_S": (B_row_t, S_cols_ref_union),
                #             "B_to_B": (B_row_t, B_cols_ref_union),
                #         }
                #         if attn_slice_old is not None:
                #             _ap_collect(
                #                 _maybe_prob(attn_slice_old),
                #                 start_idx,
                #                 end_idx,
                #                 num_heads,
                #                 which="old",
                #                 pair_masks_dict=pm,
                #                 enable_record=enable_record,
                #             )
                #         _ap_collect(
                #             _maybe_prob(attn_slice_exch),
                #             start_idx,
                #             end_idx,
                #             num_heads,
                #             which="exch",
                #             pair_masks_dict=pm,
                #             enable_record=enable_record,
                #         )
        
                # ------------------ controller ------------------
                controller_applied = False
                if i < self.heads and not ddim_inversion:
                    print("in controller")
                    attention_out = controller(attn_slice.unsqueeze(1), is_cross, place_in_unet, height, width, clip_length)
                    attn_slice = attention_out.squeeze(1)
                    controller_applied = True
        
                # # 若要記錄 "new"（controller 後）也可以：
                # if (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
                #     num_heads = self.heads
                #     for t in range(CROSS_START, F):
                #         try:
                #             S_row_t = _frame_row_mask("S", t)
                #             B_row_t = _frame_row_mask("B", t)
                #         except KeyError:
                #             continue
                #         pm = {
                #             "S_to_S": (S_row_t, S_cols_ref_union),
                #             "S_to_B": (S_row_t, B_cols_ref_union),
                #             "B_to_S": (B_row_t, S_cols_ref_union),
                #             "B_to_B": (B_row_t, B_cols_ref_union),
                #         }
                #         _ap_collect(
                #             _maybe_prob(attn_slice),
                #             start_idx,
                #             end_idx,
                #             num_heads,
                #             which="new",
                #             pair_masks_dict=pm,
                #             enable_record=enable_record,
                #         )
        
                attn_slice = attn_slice.softmax(dim=-1)



                # 2) [ADD] Cross-class attention budget cap（放在 softmax 之後、bmm 之前）
                # if DO_XFORM:
                #     ENABLE_BUDGET_CAP = True
                #     if ENABLE_BUDGET_CAP and (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
                #         alpha = 0.0  # 每個 query 對跨類的總注意力上限（可調 0.15~0.30）
                #         eps = 1e-6
                
                #         # 構造「整個序列上的 S/B 行掩碼」（把所有 clean/dirty 幀的該類像素都 OR 起來）
                #         S_rows_all = torch.zeros(sequence_length, dtype=torch.bool, device=device)
                #         B_rows_all = torch.zeros(sequence_length, dtype=torch.bool, device=device)
                #         for t in range(F):  # 也可只用 S_clean/B_clean，看你策略
                #             S_rows_all |= _frame_row_mask("S", t)
                #             B_rows_all |= _frame_row_mask("B", t)
                
                #         probs = attn_slice  # [slice, Q, K]
                
                #         # ---- S 行：限制 S→B 的總質量 ≤ α ----
                #         if S_rows_all.any():
                #             S_cross_mass = probs[:, S_rows_all][:, :, B_cols_ref_union].sum(dim=-1, keepdim=True)  # [slice, nSrows, 1]
                #             scale_S = (alpha / (S_cross_mass + eps)).clamp_max(1.0)
                #             probs[:, S_rows_all][:, :, B_cols_ref_union] *= scale_S
                
                #             # 把省下的質量回補到同類列（S→S），按原占比分配
                #             freed_S = (1.0 - scale_S) * S_cross_mass
                #             same_S = probs[:, S_rows_all][:, :, S_cols_ref_union]
                #             same_S_sum = same_S.sum(dim=-1, keepdim=True) + eps
                #             probs[:, S_rows_all][:, :, S_cols_ref_union] = same_S + freed_S * (same_S / same_S_sum)
                
                #         # ---- B 行：限制 B→S 的總質量 ≤ α ----
                #         if B_rows_all.any():
                #             B_cross_mass = probs[:, B_rows_all][:, :, S_cols_ref_union].sum(dim=-1, keepdim=True)
                #             scale_B = (alpha / (B_cross_mass + eps)).clamp_max(1.0)
                #             probs[:, B_rows_all][:, :, S_cols_ref_union] *= scale_B
                
                #             freed_B = (1.0 - scale_B) * B_cross_mass
                #             same_B = probs[:, B_rows_all][:, :, B_cols_ref_union]
                #             same_B_sum = same_B.sum(dim=-1, keepdim=True) + eps
                #             probs[:, B_rows_all][:, :, B_cols_ref_union] = same_B + freed_B * (same_B / same_B_sum)
                
                #         attn_slice = probs  # 覆寫回 attn_slice（仍是機率）

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

#slicewise
        # def _sliced_attention(query, key, value, sequence_length, dim, attention_mask, time_causal, height=None, width=None, clip_length=None):
        #     enable_record = True
        #     record_after_softmax = True  # 或你想要的預設

        #     #query (bz*heads, t x h x w, org_dim//heads )
        #     is_cross = False
        #     batch_size_attention = query.shape[0]  # bz * heads
        #     hidden_states = torch.zeros(
        #         (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        #     )
        
        #     slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        
        #     if ddim_inversion:
        #         per_frame_len = sequence_length // clip_length
        #         attention_store = torch.zeros(
        #             (batch_size_attention, clip_length, per_frame_len, per_frame_len),
        #             device=query.device,
        #             dtype=query.dtype,
        #         )
        
        #     # ===================== 基本參數/緩衝 =====================
        #     H, W, F = height, width, clip_length
        #     sp_sz = H * W
        #     assert sequence_length == F * sp_sz, "Q/K 長度必須等於 F*H*W"
        
        #     device = query.device
        #     dtype = query.dtype
        #     layer_name = place_in_unet.replace("/", "_") if isinstance(place_in_unet, str) else str(place_in_unet)
        
        #     # [NEW] 你要的設定：參考幀與交錯起點
        #     # REF_FRAMES = list(range(0, 6))  # 0..5 幀
        #     # CROSS_START = 6  # 從第 6 幀開始視為交錯段
        #     # record_after_softmax = True  # 統計 old/new 時是否記機率（維持你的原習慣）
        #     S_clean = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
        #     B_clean = [0, 1, 2, 3, 4]
        #     S_dirty = [5]
        #     B_dirty = [10,11]
            
        #     # ---- 新的觸發條件：各做各的 ----
        #     DO_XFORM_S = len(S_dirty) > 0   # A(S) 有 dirty 就處理 A
        #     DO_XFORM_B = len(B_dirty) > 0   # B 有 dirty 才處理 B
        #     DO_XFORM   = DO_XFORM_S or DO_XFORM_B
        #     print(f"DO_XFORM_S={DO_XFORM_S}, DO_XFORM_B={DO_XFORM_B}")

        #     print(f"s_clean: {S_clean}")
        #     print(f"B_clean: {B_clean}")
        #     print(f"S_dirty: {S_dirty}")
        #     print(f"B_dirty: {B_dirty}")
        #     # 全域/分層緩衝（保留你原本的）
        #     if not hasattr(self, "_ap_pairs"):
        #         self._ap_pairs = ["S_to_S", "S_to_B", "B_to_S", "B_to_B"]
        #     if not hasattr(self, "_ap_layer_stats"):
        #         self._ap_layer_stats = {}
        #     if not hasattr(self, "_ap_global_sum"):
        #         self._ap_global_sum = torch.zeros(4, dtype=torch.float32, device=device)
        #     if not hasattr(self, "_ap_global_cnt"):
        #         self._ap_global_cnt = torch.zeros(4, dtype=torch.float32, device=device)
        #     if layer_name not in self._ap_layer_stats:
        #         self._ap_layer_stats[layer_name] = {"sum": None, "cnt": None}
        
        #     # ============= S/B 的 per-frame mask 構建工具（沿用你原本的解析度索引） =============
        #     def _frame_row_mask(id_char: str, frame_idx: int) -> torch.Tensor:
        #         """Query 維度的 row mask：只開啟 frame_idx 這一幀且屬於 id_char 的像素"""
        #         id_maps = id_masks_by_res[H * W][id_char]  # [F,H,W]
        #         m = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        #         mask_hw = id_maps[frame_idx].to(device=device).reshape(-1).bool()
        #         start = frame_idx * sp_sz
        #         m[start : start + sp_sz] = mask_hw
        #         return m
        
        #     def _frame_col_mask(id_char: str, frame_idx: int) -> torch.Tensor:
        #         """Key 維度的 col mask：只開啟 frame_idx 這一幀且屬於 id_char 的像素"""
        #         # 跟 row 用同一份 id_maps（因為注意力是 Q×K，Q/K 的索引範圍一致）
        #         id_maps = id_masks_by_res[H * W][id_char]  # [F,H,W]
        #         m = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        #         mask_hw = id_maps[frame_idx].to(device=device).reshape(-1).bool()
        #         start = frame_idx * sp_sz
        #         m[start : start + sp_sz] = mask_hw
        #         return m
        
        #     # [NEW] 參考幀（0..5）之列遮罩聯集（Key 軸）
        #     # try:
        #     #     S_cols_ref_union = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        #     #     B_cols_ref_union = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        #     #     for r in REF_FRAMES:
        #     #         S_cols_ref_union |= _frame_col_mask("S", r)
        #     #         B_cols_ref_union |= _frame_col_mask("B", r)
        #     #     pair_to_idx = {p: i for i, p in enumerate(self._ap_pairs)}
        #     # except Exception as e:
        #     #     print("[AP] build ref masks failed, skip exchange:", e)
        #     #     S_cols_ref_union = B_cols_ref_union = None
        #     try:
        #         def _union_col(id_char: str, frames: list[int]) -> torch.Tensor | None:
        #             if frames is None or len(frames) == 0:
        #                 return None
        #             u = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        #             for r in frames:
        #                 u |= _frame_col_mask(id_char, r)
        #             return u
            
        #         # 4 組列遮罩 union
        #         S_cols_clean_union = _union_col("S", S_clean)
        #         S_cols_dirty_union = _union_col("S", S_dirty)
        #         B_cols_clean_union = _union_col("B", B_clean)
        #         B_cols_dirty_union = _union_col("B", B_dirty)
            
        #         pair_to_idx = {p: i for i, p in enumerate(self._ap_pairs)}
        #     except Exception as e:
        #         print("[AP] build ref masks failed:", e)
        #         S_cols_clean_union = S_cols_dirty_union = None
        #         B_cols_clean_union = B_cols_dirty_union = None    
        #     # ==== NEW: 初始化 / 蒐集工具（同層、同 head 聚合；支援 old/new 兩組） ====
        #     def _ap_init_if_needed(which: str, num_heads: int):
        #         if not hasattr(self, "_ap_pairs"):
        #             self._ap_pairs = ["S_to_S", "S_to_B", "B_to_S", "B_to_B"]
        #         if not hasattr(self, "_ap_stats"):
        #             self._ap_stats = {}
        #         if which not in self._ap_stats:
        #             self._ap_stats[which] = {
        #                 "layer": {},
        #                 "g_sum": torch.zeros(4, dtype=torch.float32, device=device),
        #                 "g_cnt": torch.zeros(4, dtype=torch.float32, device=device),
        #             }
        #         lay = self._ap_stats[which]["layer"]
        #         if layer_name not in lay or lay[layer_name]["sum"] is None:
        #             lay[layer_name] = {
        #                 "sum": torch.zeros((self.heads, 4), dtype=torch.float32, device=device),
        #                 "cnt": torch.zeros((self.heads, 4), dtype=torch.float32, device=device),
        #             }
        
        #     def _ap_collect(
        #         A_base: torch.Tensor,
        #         start_idx: int,
        #         end_idx: int,
        #         num_heads: int,
        #         which: str,
        #         pair_masks_dict: dict,
        #         enable_record: bool,
        #     ):
        #         """
        #         A_base: [slice_size, Q, K]（logits 或 prob）
        #         pair_masks_dict: {"S_to_S":(row_mask, col_mask), ...}
        #         這裡允許每次給不同的 row/col（例如每個 t）
        #         """
        #         if not enable_record:
        #             return
        #         # print("record")
        #         _ap_init_if_needed(which, num_heads)
        #         lay = self._ap_stats[which]["layer"][layer_name]
        #         g_sum = self._ap_stats[which]["g_sum"]
        #         g_cnt = self._ap_stats[which]["g_cnt"]
        
        #         global_rows = torch.arange(start_idx, end_idx, device=device)  # [slice_size]
        #         for local_j in range(A_base.shape[0]):
        #             global_idx = int(global_rows[local_j].item())
        #             head_id = global_idx % num_heads
        #             A = A_base[local_j]  # [Q,K]
        #             for pair_name, (row_mask, col_mask) in pair_masks_dict.items():
        #                 if row_mask is None or col_mask is None:
        #                     continue
        #                 sub = A[row_mask][:, col_mask]  # [nq, nk]
        #                 if sub.numel() > 0:
        #                     v = sub.mean()
        #                     idx = pair_to_idx[pair_name]
        #                     lay["sum"][head_id, idx] += v.detach().float()
        #                     lay["cnt"][head_id, idx] += 1.0
        #                     g_sum[idx] += v.detach().float()
        #                     g_cnt[idx] += 1.0
        
        #     def _maybe_prob(A: torch.Tensor):
        #         return torch.softmax(A, dim=-1) if record_after_softmax else A
        
        #     # ======= 交換工具（logits 階段） =======
        #     # [NEW]
        #     def _mk3(mask_row: torch.Tensor, mask_col: torch.Tensor, slice_len: int, Q: int, K: int):
        #         Mr = mask_row.unsqueeze(1) & torch.ones(K, dtype=torch.bool, device=mask_row.device)  # [Q,K]
        #         Mc = torch.ones(Q, dtype=torch.bool, device=mask_col.device).unsqueeze(1) & mask_col.unsqueeze(0)  # [Q,K]
        #         M = Mr & Mc
        #         return M.unsqueeze(0).expand(slice_len, -1, -1)  # [sl
        
        #     def _exchange_by_mean_shift(
        #         attn_logits: torch.Tensor,
        #         row_mask: torch.Tensor,
        #         col_mask_a: torch.Tensor,
        #         col_mask_b: torch.Tensor,
        #         strength: float = 1.0,
        #         eps: float = 1e-8,
        #     ):
        #         slice_len, Q, K = attn_logits.shape
        #         Ma = _mk3(row_mask, col_mask_a, slice_len, Q, K)
        #         Mb = _mk3(row_mask, col_mask_b, slice_len, Q, K)
            
        #         # 用 attn_logits 的 dtype 做運算以免溢出/衝突
        #         Ma_f = Ma.to(attn_logits.dtype)
        #         Mb_f = Mb.to(attn_logits.dtype)
            
        #         sum_a = (attn_logits * Ma_f).sum(dim=-1)            # [slice, Q]
        #         cnt_a = Ma_f.sum(dim=-1).clamp_min(eps)             # [slice, Q]
        #         mu_a  = sum_a / cnt_a
            
        #         sum_b = (attn_logits * Mb_f).sum(dim=-1)            # [slice, Q]
        #         cnt_b = Mb_f.sum(dim=-1).clamp_min(eps)             # [slice, Q]
        #         mu_b  = sum_b / cnt_b
            
        #         row_mask_Q = row_mask.unsqueeze(0).expand(slice_len, -1)  # [slice, Q]
        #         delta = (mu_b - mu_a) * strength
        #         delta = torch.where(row_mask_Q, delta, torch.zeros_like(delta))
            
        #         # 🔧 關鍵：把 delta cast 成 attn_logits 的 dtype
        #         delta = delta.to(attn_logits.dtype)
            
        #         # 把 delta 寫回兩塊
        #         # 用 expand_as(attn_logits) 再 masked_select，兩邊 dtype 一致
        #         attn_logits[Ma] = (attn_logits[Ma] + delta.unsqueeze(-1).expand_as(attn_logits).masked_select(Ma)).reshape(-1)
        #         attn_logits[Mb] = (attn_logits[Mb] - delta.unsqueeze(-1).expand_as(attn_logits).masked_select(Mb)).reshape(-1)

        
        #     # [NEW] 針對「單一 t 幀 vs 參考幀聯集」計算 Stos / Stob / Btos / Btob 的均值
        #     # def _pair_means_for_t(attn_logits: torch.Tensor, t: int):
        #     #     if S_cols_ref_union is None:
        #     #         return {}
        #     #     with torch.no_grad():
        #     #         S_row_t = _frame_row_mask("S", t)
        #     #         B_row_t = _frame_row_mask("B", t)
        #     #         means = {}
        #     #         # S row
        #     #         sub_SS = attn_logits[:, S_row_t, :][:, :, S_cols_ref_union]
        #     #         sub_SB = attn_logits[:, S_row_t, :][:, :, B_cols_ref_union]
        #     #         means["S_to_S"] = sub_SS.mean() if sub_SS.numel() > 0 else None
        #     #         means["S_to_B"] = sub_SB.mean() if sub_SB.numel() > 0 else None
        #     #         # B row
        #     #         sub_BS = attn_logits[:, B_row_t, :][:, :, S_cols_ref_union]
        #     #         sub_BB = attn_logits[:, B_row_t, :][:, :, B_cols_ref_union]
        #     #         means["B_to_S"] = sub_BS.mean() if sub_BS.numel() > 0 else None
        #     #         means["B_to_B"] = sub_BB.mean() if sub_BB.numel() > 0 else None
        #     #         return means, S_row_t, B_row_t
        
        #     # ===================== 主迴圈 =====================
        #     for i in range(hidden_states.shape[0] // slice_size):
        #         start_idx = i * slice_size
        #         end_idx = (i + 1) * slice_size
        
        #         query_slice = query[start_idx:end_idx]
        #         key_slice = key[start_idx:end_idx]
        
        #         if self.upcast_attention:
        #             query_slice = query_slice.float()
        #             key_slice = key_slice.float()
        
        #         attn_slice = torch.baddbmm(
        #             torch.empty(
        #                 slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device
        #             ),
        #             query_slice,
        #             key_slice.transpose(-1, -2),
        #             beta=0,
        #             alpha=self.scale,
        #         )
        
        #         if attention_mask is not None:
        #             if time_causal:
        #                 attn_slice = attn_slice + attention_mask[:1]
        #             else:
        #                 attn_slice = attn_slice + attention_mask[start_idx:end_idx]
        
        #         if self.upcast_softmax:
        #             attn_slice = attn_slice.float()
        
        #         attn_slice_old = attn_slice.clone().detach()  # [ADD] 交換前的快照 (old, logits)
        
        #         # ------------------ [MOD] 交換 + 邊際推拉 ------------------
        #         ratio_thr = 1.0
        #         strength = 2.0
        #         # RELOC_TAU = 0.25  # [ADD] post-softmax 再分配比例；若不想用，就設 0
        #         # ====== [ADD] 設定：交換成功後的「推遠/抑制」開關與強度 ======
        #         SHRINK_ON_TAIL = True         # 只在「交換成功」後才縮跨類子塊
        #         SHRINK_FACTOR  = 0.5         # η ∈ (0,1)；建議 0.2~0.5
        #         _log_eta = math.log(max(1e-8, min(0.999999, SHRINK_FACTOR)))  # logits 上的等價位移
                
        #         # ====== [ADD] logits 階段的乘法縮放：A_block <- η * A_block（以 logits 加上 log(η) 實現）======
        #         def _shrink_block(attn_logits: torch.Tensor, row_mask: torch.Tensor, col_mask: torch.Tensor):
        #             slice_len, Q, K = attn_logits.shape
        #             M = _mk3(row_mask, col_mask, slice_len, Q, K)  # [slice, Q, K], bool
        #             if M.any():
        #                 # 🔧 用 attn_logits 的 dtype/device 產生常數
        #                 _log_eta_t = attn_logits.new_tensor(_log_eta)
        #                 attn_logits[M] = attn_logits[M] + _log_eta_t

        #         # ====== S/B 髒幀分支（slice-level 平均拉扯 + 選擇性跨類縮減） ======
        #         # 參數沿用你上面定義的 ratio_thr / strength / SHRINK_ON_TAIL / _shrink_block 等
        #         # 使用「clean unions」作為正確目標
                
        #         def _means_for_S_row(attn_logits: torch.Tensor, t: int):
        #             S_row_t = _frame_row_mask("S", t)
        #             m = {}
        #             if S_cols_clean_union is not None:
        #                 sub = attn_logits[:, S_row_t, :][:, :, S_cols_clean_union]
        #                 m["S_to_S_clean"] = sub.mean() if sub.numel() > 0 else None
        #             if B_cols_clean_union is not None:
        #                 sub = attn_logits[:, S_row_t, :][:, :, B_cols_clean_union]
        #                 m["S_to_B_clean"] = sub.mean() if sub.numel() > 0 else None
        #             return m, S_row_t
                
        #         def _means_for_B_row(attn_logits: torch.Tensor, t: int):
        #             B_row_t = _frame_row_mask("B", t)
        #             m = {}
        #             if B_cols_clean_union is not None:
        #                 sub = attn_logits[:, B_row_t, :][:, :, B_cols_clean_union]
        #                 m["B_to_B_clean"] = sub.mean() if sub.numel() > 0 else None
        #             if S_cols_clean_union is not None:
        #                 sub = attn_logits[:, B_row_t, :][:, :, S_cols_clean_union]
        #                 m["B_to_S_clean"] = sub.mean() if sub.numel() > 0 else None
        #             return m, B_row_t
                
        #         # --- (A) 處理 S 髒幀：把 S_row(t) 對 B_clean 的注意力拉回 S_clean ---
        #         if DO_XFORM_S and (S_cols_clean_union is not None) and (B_cols_clean_union is not None):
        #             for t in S_dirty:
        #                 try:
        #                     means_t, S_row_t = _means_for_S_row(attn_slice, t)
        #                 except KeyError:
        #                     continue
                
        #                 exchanged_this_t = False
        #                 ms = means_t.get
        #                 if (ms("S_to_B_clean") is not None) and (ms("S_to_S_clean") is not None) and \
        #                    (means_t["S_to_B_clean"] > means_t["S_to_S_clean"] * ratio_thr):
        #                     # 拉回：在 logits 上對 (S_row_t, S_clean) 與 (S_row_t, B_clean) 做均值互換
        #                     _exchange_by_mean_shift(
        #                         attn_slice,
        #                         row_mask=S_row_t,
        #                         col_mask_a=S_cols_clean_union,  # 正確
        #                         col_mask_b=B_cols_clean_union,  # 錯誤
        #                         strength=strength,
        #                     )
        #                     exchanged_this_t = True
                
        #                 # 交換後可選：只縮跨類子塊（讓 S→B_clean 更弱）
        #                 if SHRINK_ON_TAIL and exchanged_this_t:
        #                     means_post, S_row_t2 = _means_for_S_row(attn_slice, t)
        #                     if (means_post.get("S_to_S_clean") is not None) and (means_post.get("S_to_B_clean") is not None):
        #                         if means_post["S_to_S_clean"] > means_post["S_to_B_clean"]:
        #                             _shrink_block(attn_slice, row_mask=S_row_t2, col_mask=B_cols_clean_union)
                
        #         # --- (B) 處理 B 髒幀：把 B_row(t) 對 S_clean 的注意力拉回 B_clean ---
        #         if DO_XFORM_B and (B_cols_clean_union is not None) and (S_cols_clean_union is not None):
        #             for t in B_dirty:
        #                 try:
        #                     means_t, B_row_t = _means_for_B_row(attn_slice, t)
        #                 except KeyError:
        #                     continue
                
        #                 exchanged_this_t = False
        #                 ms = means_t.get
        #                 if (ms("B_to_S_clean") is not None) and (ms("B_to_B_clean") is not None) and \
        #                    (means_t["B_to_S_clean"] > means_t["B_to_B_clean"] * ratio_thr):
        #                     _exchange_by_mean_shift(
        #                         attn_slice,
        #                         row_mask=B_row_t,
        #                         col_mask_a=B_cols_clean_union,  # 正確
        #                         col_mask_b=S_cols_clean_union,  # 錯誤
        #                         strength=strength,
        #                     )
        #                     exchanged_this_t = True
                
        #                 if SHRINK_ON_TAIL and exchanged_this_t:
        #                     means_post, B_row_t2 = _means_for_B_row(attn_slice, t)
        #                     if (means_post.get("B_to_B_clean") is not None) and (means_post.get("B_to_S_clean") is not None):
        #                         if means_post["B_to_B_clean"] > means_post["B_to_S_clean"]:
        #                             _shrink_block(attn_slice, row_mask=B_row_t2, col_mask=S_cols_clean_union)
        #         # if (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
        #         #     for t in range(CROSS_START, F):
        #         #         try:
        #         #             means_t, S_row_t, B_row_t = _pair_means_for_t(attn_slice, t)
        #         #         except KeyError:
        #         #             continue
        
        #         #         # ---- S 行：先 mean-shift 交換，再 enforce margin ----
        #         #         exchanged_this_t = False
        #         #         if (means_t.get("S_to_B") is not None) and (means_t.get("S_to_S") is not None) and \
        #         #            (means_t["S_to_B"] > means_t["S_to_S"] * ratio_thr):
        #         #             # (1) 你原本的交換：把注意力從 (S→B) 拉回 (S→S)
        #         #             _exchange_by_mean_shift(
        #         #                 attn_slice,
        #         #                 row_mask=S_row_t,
        #         #                 col_mask_a=S_cols_ref_union,  # 正確
        #         #                 col_mask_b=B_cols_ref_union,  # 錯誤
        #         #                 strength=strength,
        #         #             )
        #         #             exchanged_this_t = True
        
        #         #         if (means_t.get("B_to_S") is not None) and (means_t.get("B_to_B") is not None) and \
        #         #            (means_t["B_to_S"] > means_t["B_to_B"] * ratio_thr):
        #         #             _exchange_by_mean_shift(
        #         #                 attn_slice,
        #         #                 row_mask=B_row_t,
        #         #                 col_mask_a=B_cols_ref_union,  # 正確
        #         #                 col_mask_b=S_cols_ref_union,  # 錯誤
        #         #                 strength=strength,
        #         #             )
        #         #             exchanged_this_t = True
        #         #         # ---------- 交換（mean-shift）段落之後，加入：只縮跨類子塊 ----------
        #         #         if SHRINK_ON_TAIL and exchanged_this_t:
        #         #             print("shrink")
        #         #             # 重新用「交換後」的 attn_slice（仍是 logits）計算四對均值
        #         #             means_post, S_row_t, B_row_t = _pair_means_for_t(attn_slice, t)
                        
        #         #             # 若交換後「S→S」已經壓過「S→B」（ss > sb），就縮 S→B 子塊：A_SB <- η * A_SB
        #         #             if (means_post.get("S_to_S") is not None) and (means_post.get("S_to_B") is not None):
        #         #                 if means_post["S_to_S"] > means_post["S_to_B"]:
        #         #                     _shrink_block(attn_slice, row_mask=S_row_t, col_mask=B_cols_ref_union)
                        
        #         #             # 若交換後「B→B」已經壓過「B→S」（bb > bs），就縮 B→S 子塊：A_BS <- η * A_BS
        #         #             if (means_post.get("B_to_B") is not None) and (means_post.get("B_to_S") is not None):
        #         #                 if means_post["B_to_B"] > means_post["B_to_S"]:
        #         #                     _shrink_block(attn_slice, row_mask=B_row_t, col_mask=S_cols_ref_union)

        #         attn_slice_exch = attn_slice.clone().detach()  # [ADD] 交換後、controller 前的快照 (exch, logits)
        
        #         # ------------------ 統計蒐集：old / exch 放在 t 迴圈內逐幀收 ------------------
        #         # # [MOD]
        #         # if (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
        #         #     num_heads = self.heads
        #         #     for t in range(CROSS_START, F):
        #         #         try:
        #         #             _, S_row_t, B_row_t = _pair_means_for_t(attn_slice, t)  # 這裡只拿 mask，不重算 means
        #         #         except KeyError:
        #         #             continue
        #         #         pm = {
        #         #             "S_to_S": (S_row_t, S_cols_ref_union),
        #         #             "S_to_B": (S_row_t, B_cols_ref_union),
        #         #             "B_to_S": (B_row_t, S_cols_ref_union),
        #         #             "B_to_B": (B_row_t, B_cols_ref_union),
        #         #         }
        #         #         if attn_slice_old is not None:
        #         #             _ap_collect(
        #         #                 _maybe_prob(attn_slice_old),
        #         #                 start_idx,
        #         #                 end_idx,
        #         #                 num_heads,
        #         #                 which="old",
        #         #                 pair_masks_dict=pm,
        #         #                 enable_record=enable_record,
        #         #             )
        #         #         _ap_collect(
        #         #             _maybe_prob(attn_slice_exch),
        #         #             start_idx,
        #         #             end_idx,
        #         #             num_heads,
        #         #             which="exch",
        #         #             pair_masks_dict=pm,
        #         #             enable_record=enable_record,
        #         #         )  # [ADD]
        
        #         # ------------------ controller ------------------
        #         controller_applied = False  # [ADD]
        #         controller_outputs_prob = True  # [ADD] 若你的 controller 輸出的是概率，設 True；若輸出 logits，改成 False
        
        #         if i < self.heads and not ddim_inversion:
        #             attention_out = controller(attn_slice.unsqueeze(1), is_cross, place_in_unet, height, width, clip_length)
        #             attn_slice = attention_out.squeeze(1)
        #             controller_applied = True
        
        #         # 若要記錄 "new"（controller 後）也可以：
        #         # [ADD]
        #         # if (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
        #         #     num_heads = self.heads
        #         #     for t in range(CROSS_START, F):
        #         #         try:
        #         #             _, S_row_t, B_row_t = _pair_means_for_t(attn_slice, t)
        #         #         except KeyError:
        #         #             continue
        #         #         pm = {
        #         #             "S_to_S": (S_row_t, S_cols_ref_union),
        #         #             "S_to_B": (S_row_t, B_cols_ref_union),
        #         #             "B_to_S": (B_row_t, S_cols_ref_union),
        #         #             "B_to_B": (B_row_t, B_cols_ref_union),
        #         #         }
        #         #         _ap_collect(
        #         #             _maybe_prob(attn_slice),
        #         #             start_idx,
        #         #             end_idx,
        #         #             num_heads,
        #         #             which="new",
        #         #             pair_masks_dict=pm,
        #         #             enable_record=enable_record,
        #         #         )
        
        #         # ------------------ softmax 的正確性：只在 attn_slice 還是 logits 時才做 ------------------
        #         # [MOD]
        #         # is_logits = True
        #         # if controller_applied and controller_outputs_prob:
        #         #     is_logits = False
        #         # if is_logits:
        #         attn_slice = attn_slice.softmax(dim=-1)
        
        #         # ------------------ [ADD] Post-softmax 質量再分配（可選） ------------------
        #         # if RELOC_TAU > 0 and (S_cols_ref_union is not None) and (B_cols_ref_union is not None):
        #         #     for t in range(CROSS_START, F):
        #         #         try:
        #         #             _, S_row_t, B_row_t = _pair_means_for_t(attn_slice, t)
        #         #         except KeyError:
        #         #             continue
        #         #         # S 行：把 S→B 抽一部份 τ 給 S→S
        #         #         attn_slice = _prob_mass_realloc(attn_slice, S_row_t, S_cols_ref_union, B_cols_ref_union, tau=RELOC_TAU)
        #         #         # B 行：把 B→S 抽一部份 τ 給 B→B
        #         #         attn_slice = _prob_mass_realloc(attn_slice, B_row_t, B_cols_ref_union, S_cols_ref_union, tau=RELOC_TAU)
        
        #         attn_slice = attn_slice.to(value.dtype)
        
        #         if ddim_inversion:
        #             bz, thw, thw = attn_slice.shape
        #             t = clip_length
        #             hw = thw // t
        #             per_frame_attention = torch.empty((bz, t, hw, hw), device=attn_slice.device)
        #             for idx in range(t):
        #                 start_idx_ = idx * hw
        #                 end_idx_ = (idx + 1) * hw
        #                 per_frame_attention[:, idx, :, :] = attn_slice[:, start_idx_:end_idx_, start_idx_:end_idx_]
        #             per_frame_attention = rearrange(per_frame_attention, "b t h w -> (b t) h w")
        #             attention_store[start_idx:end_idx] = per_frame_attention
        
        #         attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
        #         hidden_states[start_idx:end_idx] = attn_slice
        
        #     if ddim_inversion:
        #         _ = controller(attention_store, is_cross, place_in_unet)
        
        #     # reshape hidden_states
        #     hidden_states = reshape_batch_dim_to_heads(hidden_states)
    

        # #     # ===================== 每次呼叫：輸出 old/new 統計 =====================
        #     # if enable_record:
        #     #     # print("record")
        #     #     try:
        #     #         outdir = "./result/Roger/*final_result/attn_record/3cls_10frame"
        #     #         os.makedirs(outdir, exist_ok=True)
        
        #     #         def _write_layer(which: str, suffix: str):
        #     #             if not hasattr(self, "_ap_stats") or which not in self._ap_stats:
        #     #                 return
        #     #             laydict = self._ap_stats[which]["layer"]
        #     #             if layer_name not in laydict:
        #     #                 return
        #     #             stat = laydict[layer_name]
        #     #             s = stat["sum"].detach().float().cpu()
        #     #             c = stat["cnt"].detach().float().cpu()
        #     #             with torch.no_grad():
        #     #                 mask = c > 0
        #     #                 mean = torch.zeros_like(s)
        #     #                 mean[mask] = s[mask] / torch.clamp(c[mask], min=1e-8)
        #     #             csv_path = os.path.join(outdir, f"{layer_name}_pairs_0to5_{suffix}.csv")  # [MOD] 檔名說明 0..5 參考
        #     #             with open(csv_path, "w", newline="", encoding="utf-8") as f:
        #     #                 writer = csv.writer(f)
        #     #                 writer.writerow(["head", "S_to_S", "S_to_B", "B_to_S", "B_to_B", "head_mean_over_pairs"])
        #     #                 for h in range(mean.shape[0]):
        #     #                     vals = mean[h].tolist()
        #     #                     head_mean = float(sum(vals) / len(vals)) if len(vals) > 0 else float("nan")
        #     #                     writer.writerow([h] + [float(x) for x in vals] + [head_mean])
        #     #             torch.save(
        #     #                 {"mean": mean, "sum": s, "cnt": c},
        #     #                 os.path.join(outdir, f"{layer_name}_pairs_0to5_{suffix}.pt"),
        #     #             )
        #     #             with open(os.path.join(outdir, f"{layer_name}_pairs_0to5_{suffix}.json"), "w", encoding="utf-8") as jf:
        #     #                 json.dump({"mean": mean.tolist()}, jf, ensure_ascii=False, indent=2)
        
        #     #         def _write_global(which: str, suffix: str):
        #     #             if not hasattr(self, "_ap_stats") or which not in self._ap_stats:
        #     #                 return
        #     #             g_sum = self._ap_stats[which]["g_sum"].detach().float().cpu()
        #     #             g_cnt = self._ap_stats[which]["g_cnt"].detach().float().cpu()
        #     #             if g_cnt.sum().item() > 0:
        #     #                 g_mean = g_sum / torch.clamp(g_cnt, min=1e-8)
        #     #                 glb = {p: float(g_mean[i].item()) for i, p in enumerate(self._ap_pairs)}
        #     #             else:
        #     #                 glb = {p: None for p in self._ap_pairs}
        #     #             with open(os.path.join(outdir, f"GLOBAL_pairs_0to5_{suffix}.json"), "w", encoding="utf-8") as f:
        #     #                 json.dump(glb, f, ensure_ascii=False, indent=2)
        
        #     #         _write_layer("old", "old")
        #     #         _write_layer("new", "new")
        #     #         _write_global("old", "old")
        #     #         _write_global("new", "new")
        
        #     #         # === [ADD] 把 exch 一起寫出 ===
        #     #         _write_layer("exch", "exch")  # <== 新增
        #     #         _write_global("exch", "exch")  # <== 新增
        #     #     except Exception as e:
        #     #         print(f"[AttnPairRecorder] write-out failed: {e}")
        
        #     return hidden_states


###original videograin sliced attention
#layer-wise
        # def _sliced_attention(query, key, value, sequence_length, dim, attention_mask, time_causal, height=None, width=None, clip_length=None):
        #     #query (bz*heads, t x h x w, org_dim//heads )
        #     # print("================== _sliced attention ====================")
        #     is_cross = False
        #     batch_size_attention = query.shape[0]   # bz * heads
        #     hidden_states = torch.zeros(
        #         (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        #     )

        #     slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]

        #     if ddim_inversion:
        #         per_frame_len = sequence_length//clip_length
        #         attention_store = torch.zeros((batch_size_attention, clip_length, per_frame_len, per_frame_len), device=query.device, dtype=query.dtype)
                
        #     # ===================== 這裡開始：紀錄設定 + 遮罩（只建一次） =====================
        #     # # # 可調設定
        #     frame_src = 9        # 人類 1-based 的「第 9 幀」
        #     frame_dst = 3        # 人類 1-based 的「第 3 幀」
        #     one_based = True     # 若你已經用 0-based，改成 False 並把上面設 8 與 2
        #     record_after_softmax = True  # True: 記機率；False: 記 logits（若 controller 已給機率就不再 softmax）
        
        #     if one_based:
        #         frame_src -= 1
        #         frame_dst -= 1
        
        #     H, W, F = height, width, clip_length
        #     sp_sz = H * W
        #     print(H,W)
        #     assert sequence_length == F * sp_sz, "Q/K 長度必須等於 F*H*W"
        
        #     device = query.device
        #     dtype = query.dtype
        #     layer_name = place_in_unet.replace("/", "_") if isinstance(place_in_unet, str) else str(place_in_unet)
        #     print(layer_name)
        #     # 初始化全域/分層緩衝（掛在 self 上，整個 run 期間累加）
        #     if not hasattr(self, "_ap_pairs"):
        #         self._ap_pairs = ["S_to_S", "S_to_B", "B_to_S", "B_to_B"]
        #     if not hasattr(self, "_ap_layer_stats"):
        #         self._ap_layer_stats = {}  # {layer_name: {"sum": tensor[H,4], "cnt": tensor[H,4]}}
        #     if not hasattr(self, "_ap_global_sum"):
        #         self._ap_global_sum = torch.zeros(4, dtype=torch.float32, device=device)
        #     if not hasattr(self, "_ap_global_cnt"):
        #         self._ap_global_cnt = torch.zeros(4, dtype=torch.float32, device=device)
        #     if layer_name not in self._ap_layer_stats:
        #         # 延後知道 head 數，用看到第一個 slice 時再建
        #         self._ap_layer_stats[layer_name] = {"sum": None, "cnt": None}
        
        #     # 取出 S/B 的 [F,H,W] -> 展平成 [F*H*W]，但只在指定幀為 True
        #     def build_seq_mask(id_char: str, frame_idx: int):
        #         # 解析度要對上當前層的 (H,W)
        #         id_maps = id_masks_by_res[H*W][id_char]  # [F,H,W]
        #         m = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        #         mask_hw = id_maps[frame_idx].to(device=device).reshape(-1).bool()
        #         start = frame_idx * sp_sz
        #         m[start:start + sp_sz] = mask_hw
        #         return m
        
        #     try:
        #         S_src_Q = build_seq_mask("S", frame_src)
        #         B_src_Q = build_seq_mask("B", frame_src)
        #         S_dst_K = build_seq_mask("S", frame_dst)
        #         B_dst_K = build_seq_mask("B", frame_dst)
        #     except KeyError as e:
        #         print("no match resolution for id mask")
        #         # 若沒對應解析度，直接跳過紀錄，但不影響前向
        #         S_src_Q = B_src_Q = S_dst_K = B_dst_K = None
        
        #     pair_masks = None
        #     if S_src_Q is not None:
        #         pair_masks = {
        #             "S_to_S": (S_src_Q, S_dst_K),
        #             "S_to_B": (S_src_Q, B_dst_K),
        #             "B_to_S": (B_src_Q, S_dst_K),
        #             "B_to_B": (B_src_Q, B_dst_K),
        #         }
        #         pair_to_idx = {p: i for i, p in enumerate(self._ap_pairs)}
        #     # ==== NEW: 初始化 / 蒐集工具（同層、同 head 聚合；支援 old/new 兩組） ====
        #     def _ap_init_if_needed(which: str, num_heads: int):
        #         # 建立容器： self._ap_stats = { which: { 'layer': {layer_name: {'sum': [H,4], 'cnt':[H,4]}},
        #         #                                         'g_sum': [4], 'g_cnt':[4] } }
        #         if not hasattr(self, "_ap_pairs"):
        #             self._ap_pairs = ["S_to_S", "S_to_B", "B_to_S", "B_to_B"]
        #         if not hasattr(self, "_ap_stats"):
        #             self._ap_stats = {}
        #         if which not in self._ap_stats:
        #             self._ap_stats[which] = {
        #                 "layer": {},
        #                 "g_sum": torch.zeros(4, dtype=torch.float32, device=device),
        #                 "g_cnt": torch.zeros(4, dtype=torch.float32, device=device),
        #             }
        #         lay = self._ap_stats[which]["layer"]
        #         if layer_name not in lay or lay[layer_name]["sum"] is None:
        #             lay[layer_name] = {
        #                 "sum": torch.zeros((num_heads, 4), dtype=torch.float32, device=device),
        #                 "cnt": torch.zeros((num_heads, 4), dtype=torch.float32, device=device),
        #             }

        #     def _ap_collect(A_base: torch.Tensor, start_idx: int, end_idx: int,
        #                     num_heads: int, which: str):
        #         """
        #         A_base: [slice_size, Q, K]（可為 logits 或 softmax 後機率，依 record_after_softmax）
        #         which: 'old' 或 'new'
        #         """
        #         _ap_init_if_needed(which, num_heads)
        #         lay = self._ap_stats[which]["layer"][layer_name]
        #         g_sum = self._ap_stats[which]["g_sum"]
        #         g_cnt = self._ap_stats[which]["g_cnt"]

        #         global_rows = torch.arange(start_idx, end_idx, device=device)  # [slice_size]
        #         for local_j in range(A_base.shape[0]):
        #             global_idx = int(global_rows[local_j].item())  # 0..(bz*heads-1)
        #             head_id = global_idx % num_heads
        #             A = A_base[local_j]  # [Q,K]

        #             for pair_name, (row_mask, col_mask) in pair_masks.items():
        #                 sub = A[row_mask][:, col_mask]  # [nq, nk]
        #                 if sub.numel() > 0:
        #                     v = sub.mean()
        #                     idx = pair_to_idx[pair_name]
        #                     lay["sum"][head_id, idx] += v.detach().float()
        #                     lay["cnt"][head_id, idx] += 1.0
        #                     g_sum[idx] += v.detach().float()
        #                     g_cnt[idx] += 1.0

        #     def _maybe_prob(A: torch.Tensor):
        #         return torch.softmax(A, dim=-1) if record_after_softmax else A
        #     # ======= 交換工具（logits 階段） =======  # [NEW]
        #     # [NEW] 生成 3D 掩碼 [slice, Q, K] 的工具
        #     def _mk3(mask_row: torch.Tensor, mask_col: torch.Tensor, slice_len: int, Q: int, K: int):
        #         # mask_row: [Q] bool, mask_col: [K] bool
        #         Mr = mask_row.unsqueeze(1) & torch.ones(K, dtype=torch.bool, device=mask_row.device)  # [Q,K] with row mask
        #         Mc = torch.ones(Q, dtype=torch.bool, device=mask_col.device).unsqueeze(1) & mask_col.unsqueeze(0)  # [Q,K] with col mask
        #         M = Mr & Mc  # [Q,K]
        #         return M.unsqueeze(0).expand(slice_len, -1, -1)  # [slice, Q, K]
            
        #     # [NEW] 平均值交換：對兩塊做「均值互換/靠攏」，不要求塊形狀相同
        #     def _exchange_by_mean_shift(attn_logits: torch.Tensor,
        #                                 row_mask: torch.Tensor,
        #                                 col_mask_a: torch.Tensor,
        #                                 col_mask_b: torch.Tensor,
        #                                 strength: float = 1.0,
        #                                 eps: float = 1e-8):
        #         """
        #         對 attn_logits 的兩個子區塊 A(row_mask,col_mask_a) 與 B(row_mask,col_mask_b)
        #         進行「均值交換/靠攏」：
        #           A <- A + strength * (μ_B - μ_A)
        #           B <- B + strength * (μ_A - μ_B)
        #         注意：這是 logits 平移，不改變兩塊內部相對結構；softmax 之後會自動正規化。
        #         """
        #         slice_len, Q, K = attn_logits.shape
        #         Ma = _mk3(row_mask, col_mask_a, slice_len, Q, K)  # [slice, Q, K]
        #         Mb = _mk3(row_mask, col_mask_b, slice_len, Q, K)
            
        #         Ma_f = Ma.float()
        #         Mb_f = Mb.float()
            
        #         # per-row（在選定的 rows 上）計算兩塊的平均
        #         # 形狀：row-mean 是 [slice, Q]，未被選中的 row 會得到 0/NaN，因此再乘 row_mask 遮掉
        #         sum_a = (attn_logits * Ma_f).sum(dim=-1)          # [slice, Q]
        #         cnt_a = Ma_f.sum(dim=-1).clamp_min(eps)           # [slice, Q]
        #         mu_a  = sum_a / cnt_a                             # [slice, Q]
            
        #         sum_b = (attn_logits * Mb_f).sum(dim=-1)          # [slice, Q]
        #         cnt_b = Mb_f.sum(dim=-1).clamp_min(eps)           # [slice, Q]
        #         mu_b  = sum_b / cnt_b                             # [slice, Q]
            
        #         # 只對 row_mask 為真的行生效
        #         row_mask_Q = row_mask.unsqueeze(0).expand(slice_len, -1)  # [slice, Q]
        #         delta = (mu_b - mu_a) * strength                          # [slice, Q]
        #         delta = torch.where(row_mask_Q, delta, torch.zeros_like(delta))  # 其他行不動
            
        #         # 把 delta 套回兩個塊（沿列擴展到列中的每個 K）
        #         attn_logits[Ma] = (attn_logits[Ma] + delta.unsqueeze(-1).expand_as(attn_logits).masked_select(Ma)).reshape(-1)
        #         attn_logits[Mb] = (attn_logits[Mb] - delta.unsqueeze(-1).expand_as(attn_logits).masked_select(Mb)).reshape(-1)

        
        #     def _pair_means(attn_logits: torch.Tensor):  # [NEW]
        #         if pair_masks is None:
        #             return {}
        #         with torch.no_grad():
        #             means = {}
        #             for name, (row_mask, col_mask) in pair_masks.items():
        #                 sub = attn_logits[:, row_mask, :][:, :, col_mask]
        #                 means[name] = sub.mean() if sub.numel() > 0 else None
        #             return means
        #     # ===================== 主迴圈 =====================
        #     for i in range(hidden_states.shape[0] // slice_size):
        #         start_idx = i * slice_size
        #         end_idx = (i + 1) * slice_size

        #         query_slice = query[start_idx:end_idx]
        #         key_slice = key[start_idx:end_idx]

        #         if self.upcast_attention:
        #             query_slice = query_slice.float()
        #             key_slice = key_slice.float()

        #         attn_slice = torch.baddbmm(
        #             torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
        #             query_slice,
        #             key_slice.transpose(-1, -2),
        #             beta=0,
        #             alpha=self.scale,
        #         )

        #         if attention_mask is not None:
        #             if time_causal:
        #                 attn_slice = attn_slice + attention_mask[:1]
        #             else:
        #                 attn_slice = attn_slice + attention_mask[start_idx:end_idx]

        #         if self.upcast_softmax:
        #             attn_slice = attn_slice.float()
                
        #         attn_slice_old = attn_slice.clone().detach()
        #         # if i < self.heads:
        #         if not ddim_inversion:
        #             # print("controller:", type(controller))
        #             attention_probs = controller((attn_slice.unsqueeze(1)),is_cross, place_in_unet, height, width, clip_length)
        #             attn_slice = attention_probs.squeeze(1)
        #         # ===== 交換守門員（logits 階段、softmax 前） =====  # [NEW]
        #         if pair_masks is not None:
        #             means = _pair_means(attn_slice)
        #             # 觸發條件（你可調整）
        #             ratio_thr = 1.2
        #             # gap_thr   = 0.05
        
        #             # S 行：若 S→B 顯著大於 S→S，就互換兩塊
        #             if (means.get("S_to_B") is not None) and (means.get("S_to_S") is not None) and (means["S_to_B"] > means["S_to_S"] * ratio_thr):
        #                 print("swap")
        #                 # and (means["S_to_B"] - means["S_to_S"]) > gap_thr):
        #                 _exchange_by_mean_shift(attn_slice,
        #                                         pair_masks["S_to_S"][0],  # row_mask: S_src_Q
        #                                         pair_masks["S_to_S"][1],  # col_mask_a: S_dst_K
        #                                         pair_masks["S_to_B"][1],  # col_mask_b: B_dst_K
        #                                         strength=1.0)
        
        #             # B 行：若 B→S 顯著大於 B→B，就互換兩塊
        #             if (means.get("B_to_S") is not None) and (means.get("B_to_B") is not None) and (means["B_to_S"] > means["B_to_B"] * ratio_thr):
        #                 print("swap")
        #                 # and (means["B_to_S"] - means["B_to_B"]) > gap_thr):
        #                 _exchange_by_mean_shift(attn_slice,
        #                                         pair_masks["B_to_B"][0],  # row_mask: B_src_Q
        #                                         pair_masks["B_to_B"][1],  # col_mask_a: B_dst_K
        #                                         pair_masks["B_to_S"][1],  # col_mask_b: S_dst_K
        #                                         strength=1.0)
        #         # delta = (attn_slice - attn_slice_old).abs().max().item()
        #         # print(f"[DEBUG] slice {i}: max|Δ| = {delta:.20e}")
        #         # if delta <= 1e-20:
        #         #     print(f"[DEBUG] controller 幾乎沒改變 (≤1e-20)")
        #         # ====== 蒐集配對（OLD：controller 之前） ======
        #         if pair_masks is not None:
        #             num_heads = self.heads
        #             if attn_slice_old is not None:
        #                 _ap_collect(_maybe_prob(attn_slice_old), start_idx, end_idx, num_heads, which="old")
        #         # ====== 蒐集配對（NEW：controller 之後）======
        #         if pair_masks is not None:
        #             num_heads = self.heads
        #             _ap_collect(_maybe_prob(attn_slice), start_idx, end_idx, num_heads, which="new")
          
                
        #         attn_slice = attn_slice.softmax(dim=-1)

        #         # cast back to the original dtype
        #         attn_slice = attn_slice.to(value.dtype)
        #         ## bz == 1, sliced head 
        #         if ddim_inversion:
        #             # attn_slice (1, thw, thw)
        #             bz, thw, thw = attn_slice.shape
        #             t = clip_length
        #             hw =  thw // t
        #             # 初始化 per_frame_attention
        #             # (1, t, hxw)

        #             per_frame_attention = torch.empty((bz, t, hw, hw), device=attn_slice.device)

        #             # # 循环提取每一帧的对角线注意力
        #             for idx in range(t):
        #                 start_idx_ = idx * hw
        #                 end_idx_ = (idx + 1) * hw
        #                 # per frame attention extraction
        #                 per_frame_attention[:, idx, :, :] = attn_slice[:, start_idx_:end_idx_, start_idx_:end_idx_]

        #                 # current_query_block = attn_slice[:, start_idx_:end_idx_, :] 
        #                 # aggregated_attention = current_query_block.view(bz, hw, t, hw).mean(dim=2)
        #                 # # print('aggregated_attention',aggregated_attention.shape)
        #                 # per_frame_attention[:, idx, :, :] = aggregated_attention

        #             per_frame_attention = rearrange(per_frame_attention, "b t h w -> (b t) h w")
        #             attention_store[start_idx:end_idx] = per_frame_attention
                
        #         attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

        #         hidden_states[start_idx:end_idx] = attn_slice
        #     if ddim_inversion:
        #         # attention store (bz*heads, t , h, w) h=res, w=res
        #         _ = controller(attention_store, is_cross, place_in_unet)

        #     # reshape hidden_states
        #     hidden_states = reshape_batch_dim_to_heads(hidden_states)

        #     # ===================== 每次呼叫時：輸出當前層 + 全域（old/new 各一份） =====================
        #     try:
        #         outdir = "./result/Roger/debug_attn_pairs_Roger_new"
        #         os.makedirs(outdir, exist_ok=True)

        #         def _write_layer(which: str, suffix: str):
        #             if not hasattr(self, "_ap_stats") or which not in self._ap_stats:
        #                 return
        #             laydict = self._ap_stats[which]["layer"]
        #             if layer_name not in laydict:
        #                 return
        #             stat = laydict[layer_name]
        #             s = stat["sum"].detach().float().cpu()
        #             c = stat["cnt"].detach().float().cpu()
        #             with torch.no_grad():
        #                 mask = c > 0
        #                 mean = torch.zeros_like(s)
        #                 mean[mask] = s[mask] / torch.clamp(c[mask], min=1e-8)

        #             csv_path = os.path.join(outdir, f"{layer_name}_pairs_9to3_{suffix}.csv")
        #             with open(csv_path, "w", newline="", encoding="utf-8") as f:
        #                 writer = csv.writer(f)
        #                 writer.writerow(["head", "S_to_S", "S_to_B", "B_to_S", "B_to_B", "head_mean_over_pairs"])
        #                 for h in range(mean.shape[0]):
        #                     vals = mean[h].tolist()
        #                     head_mean = float(sum(vals) / len(vals)) if len(vals) > 0 else float("nan")
        #                     writer.writerow([h] + [float(x) for x in vals] + [head_mean])

        #             torch.save({"mean": mean, "sum": s, "cnt": c},
        #                        os.path.join(outdir, f"{layer_name}_pairs_9to3_{suffix}.pt"))
        #             with open(os.path.join(outdir, f"{layer_name}_pairs_9to3_{suffix}.json"), "w", encoding="utf-8") as jf:
        #                 json.dump({"mean": mean.tolist()}, jf, ensure_ascii=False, indent=2)

        #         def _write_global(which: str, suffix: str):
        #             if not hasattr(self, "_ap_stats") or which not in self._ap_stats:
        #                 return
        #             g_sum = self._ap_stats[which]["g_sum"].detach().float().cpu()
        #             g_cnt = self._ap_stats[which]["g_cnt"].detach().float().cpu()
        #             if g_cnt.sum().item() > 0:
        #                 g_mean = g_sum / torch.clamp(g_cnt, min=1e-8)
        #                 glb = {p: float(g_mean[i].item()) for i, p in enumerate(self._ap_pairs)}
        #             else:
        #                 glb = {p: None for p in self._ap_pairs}
        #             with open(os.path.join(outdir, f"GLOBAL_pairs_9to3_{suffix}.json"), "w", encoding="utf-8") as f:
        #                 json.dump(glb, f, ensure_ascii=False, indent=2)

        #         # 分別輸出 old / new
        #         _write_layer("old", "old")
        #         _write_layer("new", "new")
        #         _write_global("old", "old")
        #         _write_global("new", "new")

        #     except Exception as e:
        #         print(f"[AttnPairRecorder] write-out failed: {e}")
        
        #     # ===================== 返回原本 hidden_states（你原本的流程照常） =====================
        #     return hidden_states
        
      
        # def fully_frame_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, clip_length=None, inter_frame=False, time_causal=True, **kwargs):
        #     batch_size, sequence_length, _ = hidden_states.shape
        #     # print("hidden_states.shape",hidden_states.shape)
        #     # print("sequence_length",sequence_length)

        #     encoder_hidden_states = encoder_hidden_states
        #     h = kwargs['height']
        #     w = kwargs['width']
        #     if self.group_norm is not None:
        #         hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        #     query = self.to_q(hidden_states)  # (bf) x d(hw) x c
        #     self.q = query
        #     if self.inject_q is not None:
        #         query = self.inject_q
        #     dim = query.shape[-1]
        #     query_old = query.clone()

        #     # All frames
        #     #init query (bz*t, hxw, dim)
        #     query = rearrange(query, "(b f) d c -> b (f d) c", f=clip_length)
        #     query = reshape_heads_to_batch_dim(query)  #(bz*heads, txhxw, dim//heads)

        #     if self.added_kv_proj_dim is not None:
        #         raise NotImplementedError

        #     encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        #     key = self.to_k(encoder_hidden_states)
        #     self.k = key
        #     if self.inject_k is not None:
        #         key = self.inject_k
        #     key_old = key.clone()
        #     value = self.to_v(encoder_hidden_states)

        #     if inter_frame:
        #         key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
        #         value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
        #         key = rearrange(key, "b f d c -> b (f d) c",)
        #         value = rearrange(value, "b f d c -> b (f d) c")
        #     else:
        #         # All frames
        #         key = rearrange(key, "(b f) d c -> b (f d) c", f=clip_length)
        #         value = rearrange(value, "(b f) d c -> b (f d) c", f=clip_length)

        #     key = reshape_heads_to_batch_dim(key)
        #     value = reshape_heads_to_batch_dim(value)

        #     if attention_mask is not None:
        #         if attention_mask.shape[-1] != query.shape[1]:
        #             target_length = query.shape[1]
        #             attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
        #             attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        #     #print("query.shape[0]",query.shape[0])  # 16
            # self._slice_size = 1   ### 8
            # sequence_length_full_frame = query.shape[1]

            # # attention, what we cannot get enough of
            # if self._use_memory_efficient_attention_xformers and query.shape[-2] > clip_length*(32 ** 2):
            #     hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask)
            #     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            #     hidden_states = hidden_states.to(query.dtype)
                
            # else:
            #     # if ddim_inversion:
            #     # #if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            #     #     hidden_states = _attention(query, key, value, attention_mask)
            #     # else:
            #     attn_mask_local = None
            #     if time_causal:
            #         Q = sequence_length_full_frame  # = query.shape[-2]
            #         attn_mask_local = build_frame_causal_mask(
            #             sequence_length=Q,
            #             clip_length=clip_length,
            #             device=query.device,
            #             dtype=query.dtype,
            #             include_same_frame=True,
            #         )
            #     hidden_states = _sliced_attention(
            #         query,
            #         key,
            #         value,
            #         sequence_length=sequence_length_full_frame,  # 通常是 clip_length * H * W
            #         dim=dim,
            #         attention_mask=attn_mask_local,
            #         time_causal=time_causal,
            #         clip_length=clip_length,    # 例如 8 幀
            #         height=h,                   # 每幀高
            #         width=w,                    # 每幀寬
                # )
                # hidden_states = _sliced_attention(query, key, value, sequence_length_full_frame, dim, attention_mask, time_causal)


#         def fully_frame_forward(
#             hidden_states,
#             encoder_hidden_states=None,
#             attention_mask=None,
#             clip_length=None,
#             inter_frame=False,
#             flow_only=True,
#             time_causal=False,
#             **kwargs
#         ):
#             # ==========================
#             # 前置：shape & 參數
#             # ==========================
#             batch_size, sequence_length, _ = hidden_states.shape
#             h = kwargs['height']
#             w = kwargs['width']
#             use_fullframe_layer = False
#             device = hidden_states.device
#             N = h * w
#             dim_out = hidden_states.shape[-1]
        
#             # [MOD-BEGIN] 保存「原始輸入 hidden_states」供後續幀級拼接用（與 to_out 後維度一致）
#             hidden_in_tokens = hidden_states.clone()  # (B*F, N, C)
#             # [MOD-END]
        
#             # group norm（若有）
#             if self.group_norm is not None:
#                 hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
#             # q / k / v（full-frame block 的輸入）
#             query = self.to_q(hidden_states)  # (B*F, N, Cq)
#             self.q = query
#             if self.inject_q is not None:
#                 query = self.inject_q
        
#             dim = query.shape[-1]
#             query_old = query.clone()
        
#             # qq -> (B, F*N, Cq) -> heads batch 化
#             query = rearrange(query, "(b f) d c -> b (f d) c", f=clip_length)
#             query = reshape_heads_to_batch_dim(query)  # (B*heads, F*N, Cq//heads)
        
#             if self.added_kv_proj_dim is not None:
#                 raise NotImplementedError
#             if encoder_hidden_states is None:
#                 print("encoder hidden state is none.")
#             encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
#             key = self.to_k(encoder_hidden_states); self.k = key
#             if self.inject_k is not None:
#                 key = self.inject_k
#             key_old = key.clone()
        
#             value = self.to_v(encoder_hidden_states)
#             value_old = value.clone()
        
#             # full-frame 的 key/value 展平到 (B, F*N, C)
#             if use_fullframe_layer:
#                 if inter_frame:
#                     key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
#                     value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
#                     key = rearrange(key, "b f d c -> b (f d) c")
#                     value = rearrange(value, "b f d c -> b (f d) c")
#                 else:
#                     key = rearrange(key, "(b f) d c -> b (f d) c", f=clip_length)
#                     value = rearrange(value, "(b f) d c -> b (f d) c", f=clip_length)
#                 key = reshape_heads_to_batch_dim(key)
#                 value = reshape_heads_to_batch_dim(value)
        
#                 # attention mask 準備
#                 if attention_mask is not None:
#                     if attention_mask.shape[-1] != query.shape[1]:
#                         target_length = query.shape[1]
#                         attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
#                         attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
            
#                 self._slice_size = 1
#                 sequence_length_full_frame = query.shape[1]
            
#                 # ==========================
#                 # Full-frame attention 前向（得到 ff_proj）
#                 # ==========================
#                 if self._use_memory_efficient_attention_xformers and query.shape[-2] > clip_length*(32 ** 2):
#                     hidden_ff = _memory_efficient_attention_xformers(query, key, value, attention_mask, time_causal=time_causal)
#                     hidden_ff = hidden_ff.to(query.dtype)
#                     ff_tokens_all = hidden_ff
#                 else:
#                     attn_mask_local = None
#                     if time_causal:
#                         Q = sequence_length_full_frame
#                         attn_mask_local = build_frame_causal_mask(
#                             sequence_length=Q,
#                             clip_length=clip_length,
#                             device=query.device,
#                             dtype=query.dtype,
#                             include_same_frame=True,
#                         )
#                     hidden_ff = _sliced_attention(
#                         query, key, value, sequence_length_full_frame, dim, attn_mask_local, time_causal, h, w, clip_length
#                     )
#                     ff_tokens_all = hidden_ff
            
#                 # to_out（把 full-frame 的輸出投影回 token 維度）
#                 ff_proj = self.to_out[1]( self.to_out[0](ff_tokens_all) )  # (B, F*N, C) after reshape later
            
#                 # ==========================
#                 # [MOD-BEGIN] 幀級拼接：用於「進入 flow 前」的輸入
#                 #   1) 取得 純 full-frame 的前 K 幀（ff_front）
#                 #   2) 取得 原始輸入 hidden 的後半幀（orig_tail）
#                 #   3) 拼成 flow 的輸入序列（splice_for_flow）
#                 # ==========================
#                 ff_frames = int(kwargs.get("ff_frames", 6))   # 預設 0..5 幀
#                 # 還原 b×(F*N)×C -> b×F×N×C
#                 ff_grid_all   = rearrange(ff_proj, "b (f n) c -> b f n c", f=clip_length, n=N)
#                 hidden_in_grid= rearrange(hidden_in_tokens, "(b f) n c -> b f n c", f=clip_length)
            
#                 ff_front   = ff_grid_all[:, :ff_frames]          # b×K×N×C（純 full-frame 的 0..K-1 幀）
#                 orig_tail  = hidden_in_grid[:, ff_frames:]       # b×(F-K)×N×C（原始輸入的 K..F-1 幀）
            
#                 splice_for_flow = torch.cat([ff_front, orig_tail], dim=1)  # b×F×N×C
#                 # 轉成 (B*F, N, C) 交給 flow 分支用
#                 encoder_hidden_states_for_flow = rearrange(splice_for_flow, "b f n c -> (b f) n c")
#                 # [MOD-END]
            
#                 # ==========================
#                 # Flow（semantic / original）前向
#                 # ==========================
#                 flow_proj_exists = False
            
#             # if controller.__class__.__name__ == "ST_Layout_Attn_ControlEdit" and ((h != 64) and (w != 64)):
#             #     if [h, w] in kwargs['flatten_res']:
#             #         # ------ 前置：依你原本的 old_qk/flow_only 分支 ------
#             #         if kwargs.get("old_qk", 1) == 1:
#             #             print("oldqk")
#             #             query_old_ff = query_old
#             #             key_old_ff   = key_old
#             #         else:
#             #             print("no oldqk")
#             #             query_old_ff = encoder_hidden_states_for_flow
#             #             key_old_ff   = encoder_hidden_states_for_flow
        
#             #         # if use_fullframe_layer:
#             #         #     value_old_ff = ff_proj  # 注意：ff_proj 是 (b, F*N, C)，與下方 reshape 對齊
#             #         # else:
#             #         #     value_old_ff = value_old
        
#             #         # 做成 [B,F,H,W,D] 以便沿軌跡 gather
#             #         Bsmall = batch_size // clip_length
#             #         _key_ff = rearrange(key_old_ff, '(b f) (hh ww) d -> b f hh ww d', b=Bsmall, f=clip_length, hh=h, ww=w)
        
#             #         # [MOD-BEGIN] value_for_gather 改用「拼接後的 flow 輸入」對齊
#             #         value_for_gather = encoder_hidden_states_for_flow if use_fullframe_layer else value_old_ff
#             #         _value_ff = rearrange(value_for_gather, '(b f) (hh ww) d -> b f hh ww d', b=Bsmall, f=clip_length, hh=h, ww=w)
#             #         # [MOD-END]
        
#             #         # traj/mask 正規化（沿用你原本流程）
#             #         traj_in = kwargs["traj"]
#             #         mask_in = kwargs["mask"]
#             #         traj_ff, mask_ff = normalize_traj_and_mask(traj_in, mask_in, F_clip=clip_length, N=N)
#             #         if traj_ff.size(2) >= clip_length:
#             #             traj_ff = torch.cat([traj_ff[:, :, 0:1, :], traj_ff[:, :, -clip_length+1:, :]], dim=2)
#             #             mask_ff = torch.cat([mask_ff[:, :, 0:1],     mask_ff[:, :, -clip_length+1:]],     dim=2)
        
#             #         # flow_only 開關（沿用你原先設計）
#             #         if use_fullframe_layer:
#             #             flow_only = False
#             #         else:
#             #             flow_only = True
        
#             #         with torch.no_grad():
#             #             hidden_states_ff_flow = flow_semantic_traj_attention(
#             #                 query_old=query_old_ff,
#             #                 key_old=key_old_ff,
#             #                 # value_old=value_old,
#             #                 encoder_hidden_states=encoder_hidden_states_for_flow,  # [MOD] ← 用幀級拼接後的序列
#             #                 group_norm=self.group_norm,
#             #                 traj=traj_ff, mask=mask_ff,
#             #                 time_causal=time_causal,
#             #                 _key=_key_ff, _value=_value_ff,
#             #                 h=h, w=w, clip_length=clip_length,
#             #                 heads=self.heads, controller=controller,
#             #                 sem_chunk=getattr(self, "sem_chunk", 64),
#             #                 use_sem_aug=getattr(self, "use_sem_aug", True),
#             #                 flow_only=flow_only,
#             #                 old_qk=kwargs.get("old_qk", 1),
#             #             )
        
#             #         # to_out（flow）
#             #         hidden_states_ff_flow = self.to_out[0](hidden_states_ff_flow)
#             #         hidden_states_ff_flow = self.to_out[1](hidden_states_ff_flow)
        
#             #         flow_proj = hidden_states_ff_flow  # b×(F*N)×C 之後會 reshape
#             #         flow_proj_exists = True
        
#             # else:
#                     # ======= 原版「非 semantic 控制」分支（保留） =======
#             print(f"--- [h,w] : {[h,w]}, {kwargs['flatten_res']}")
#             if [h,w] in kwargs['flatten_res'] :
#                 use_fullframe_layer = True
#                 print("--- start flow guided attention ---")
#                 # if not flow_only:
#                 # encoder_hidden_states = encoder_hidden_states_for_flow
#                 if self.group_norm is not None:
#                     encoder_hidden_states = self.group_norm(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
    
#                 if kwargs["old_qk"] == 1:
#                     # print("--- old query and key for attention ---")
#                     query = query_old
#                     key = key_old
#                     # print(f"flow query : {query.shape}") # [30, 4096, 320]
#                     # print(f"flow key : {key.shape}") # [30, 4096, 320]
#                 # else:
#                 #     # print("--- hiden state for attention ---")
#                 # query = encoder_hidden_states_for_flow
#                 # key = encoder_hidden_states_for_flow
#                 # value = hidden_states
#                 # if not flow_only or use_fullframe_layer:
              
# #######################################################################################
#                 # if use_fullframe_layer:
#                 #     # print("ababababababababababababababababababababa")
#                 #     value = encoder_hidden_states # [30, 4096, 320]
#                 # else:
#                 value = value_old  # 直接沿用
#                 # value = value_old  # 直接沿用
# #######################################################################################
#                 traj = kwargs["traj"]
#                 traj = rearrange(traj, '(f n) l d -> f n l d', f=clip_length, n=sequence_length)
                
#                 mask = rearrange(kwargs["mask"], '(f n) l -> f n l', f=clip_length, n=sequence_length)
#                 mask = torch.cat([mask[:, :, 0].unsqueeze(-1), mask[:, :, -clip_length+1:]], dim=-1)
#                 # print(f"--- traj shape is {traj.shape} ---")  # traj shape is torch.Size([15, 4096, 39, 3])
#                 # print(f"--- mask shape is {mask.shape} ---")  # mask shape is torch.Size([15, 4096, 15])
#                 #print('traj',traj.shape)
#                 #print('mask',mask.shape)
    
#                 traj_key_sequence_inds = torch.cat([traj[:, :, 0, :].unsqueeze(-2), traj[:, :, -clip_length+1:, :]], dim=-2)
#                 t_inds = traj_key_sequence_inds[:, :, :, 0]
#                 x_inds = traj_key_sequence_inds[:, :, :, 1]
#                 y_inds = traj_key_sequence_inds[:, :, :, 2]
    
#                 ##### attention modification for previous frames #################################
#                 anchor = t_inds[:, :, 0].unsqueeze(-1).expand_as(t_inds)
    
#                 if time_causal:
#                     traj_mask = t_inds <= anchor   
#                 else:
#                     traj_mask = torch.ones_like(t_inds, dtype=torch.bool)
#                 t_inds = torch.where(traj_mask, t_inds, torch.zeros_like(t_inds))
#                 x_inds = torch.where(traj_mask, x_inds, torch.zeros_like(x_inds))
#                 y_inds = torch.where(traj_mask, y_inds, torch.zeros_like(y_inds))
#                 #############################################################
    
                
#                 # for i in range(14):
#                 #     print(f"--- tinds : {t_inds.shape}, {t_inds[i, 2886]}") # (15, 4096, 15)
#                 #     print(f"--- xinds : {x_inds.shape}, {x_inds[i, 2886]}")
#                 #     print(f"--- yinds : {y_inds.shape}, {y_inds[i, 2886]}")
#                 #     print("-" * 25)
#                 # for j in range(2000, 2020, 1):
#                 #     print(f"--- tinds : {t_inds.shape}, {t_inds[7, j]}") # (15, 4096, 15)
#                 #     print(f"--- xinds : {x_inds.shape}, {x_inds[7, j]}")
#                 #     print(f"--- yinds : {y_inds.shape}, {y_inds[7, j]}")
#                 #     print("+" * 25)
#                 query_tempo = query.unsqueeze(-2)   
#                 # print(f"--- query tempo shape: {query_tempo.shape} ---") # (2*15, 4096, 1, 320)
#                 _key = rearrange(key, '(b f) (h w) d -> b f h w d', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
#                 _value = rearrange(value, '(b f) (h w) d -> b f h w d', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
#                 # print(f"--- _key shape: {_key.shape} ---") # [2, 15, 64, 64, 320]
#                 # print(f"--- _value shape: {_value.shape} ---") # [2, 15, 64, 64, 320]
#                 key_tempo = _key[:, t_inds, x_inds, y_inds] #  [2, 15, 4096, 15, 320])
#                 value_tempo = _value[:, t_inds, x_inds, y_inds] # [2, 15, 4096, 15, 320])
#                 # print(f"--- key tempo shape: {key_tempo.shape} ---")
#                 # print(f"--- value tempo shape: {value_tempo.shape} ---")
#                 key_tempo = rearrange(key_tempo, 'b f n l d -> (b f) n l d') # [30, 4096, 15, 320]
#                 value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d') # [30, 4096, 15, 320]
#                 # print(f"--- key tempo shape: {key_tempo.shape} ---")
#                 # print(f"--- value tempo shape: {value_tempo.shape} ---")    
          
#                 ##### attention modification#################################
#                 seq_mask = mask
#                 # print(f"--- original mask shape : {seq_mask.shape}")
#                 # print(f"--- traj_mask shape : {traj_mask.shape}")
#                 keep_mask = seq_mask & traj_mask
#                 mask = rearrange(torch.stack([keep_mask, keep_mask]),  'b f n l -> (b f) n l')
    
#                 #############################################################
#                 # mask = rearrange(torch.stack([mask, mask]),  'b f n l -> (b f) n l')
#                 mask = mask[:,None].repeat(1, self.heads, 1, 1).unsqueeze(-2)
    
                
#                 attn_bias = torch.zeros_like(mask, dtype=key_tempo.dtype) # regular zeros_like
#                 attn_bias[~mask] = -torch.inf  
    
#                 # print('attn_bias',attn_bias.shape)  (30, H, 1, 4096, 15)
#                 # print('query_tempo',query_tempo.shape) # query_tempo torch.Size([30, 4096, 1, 320])
#                 # print('key_tempo',key_tempo.shape)     # key_tempo torch.Size([30, 4096, 15, 320])
#                 # print('value_tempo',value_tempo.shape) # value_tempo torch.Size([30, 4096, 15, 320])
#                 # flow attention
#                 query_tempo = reshape_heads_to_batch_dim3(query_tempo) # query_tempo torch.Size([30, 8, 4096, 1, 40])
#                 key_tempo = reshape_heads_to_batch_dim3(key_tempo)     # key_tempo torch.Size([30, 8, 4096, 15, 40])
#                 value_tempo = reshape_heads_to_batch_dim3(value_tempo) # value_tempo torch.Size([30, 8, 4096, 15, 40])
#                 # print("-------------------------------------------")
#                 # print('query_tempo',query_tempo.shape)
#                 # print('key_tempo',key_tempo.shape)
#                 # print('value_tempo',value_tempo.shape)
                
#                 attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(query_tempo.size(-1)) + attn_bias
#                 attn_matrix2 = F.softmax(attn_matrix2, dim=-1)
#                 out = (attn_matrix2@value_tempo).squeeze(-2)
    
#                 hidden_states_plain = rearrange(out,'(b f) k (h w) d -> b (f h w) (k d)', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
    
#                 # linear proj
#                 hidden_states_plain = self.to_out[0](hidden_states_plain)
#                 # print("decrease self attention.")
#                 # CA_ALPHA = 0.6 # 0.1~0.3 常見；想再弱就更小
#                 # hidden_states = hidden_states * CA_ALPHA
#                 # dropout
#                 hidden_states_plain = self.to_out[1](hidden_states_plain)


    
#                 flow_proj = hidden_states_plain
#                 flow_proj_exists = True
#                 return rearrange(flow_proj, "b (f n) c -> (b f) n c", f=clip_length, n=N)
#             # ==========================
#             # 最終輸出：前 K 幀用純 full-frame；其餘幀用 flow 結果
#             # ==========================
#             if not flow_proj_exists:
#                 # 沒做 flow：直接回 full-frame 結果
#                 print("no flow proj exists")
#                 return rearrange(ff_proj, "b (f n) c -> (b f) n c", f=clip_length, n=N)
        
#             ff_grid   = rearrange(ff_proj,   "b (f n) c -> b f n c", f=clip_length, n=N)
#             flow_grid = rearrange(flow_proj, "b (f n) c -> b f n c", f=clip_length, n=N)
        
#             frame_mask = torch.arange(clip_length, device=device) < ff_frames   # [F]
#             frame_mask = frame_mask.view(1, clip_length, 1, 1).to(ff_grid.dtype)
        
#             mixed_grid = frame_mask * ff_grid + (1.0 - frame_mask) * flow_grid  # b×F×N×C
#             mixed_tokens = rearrange(mixed_grid, "b f n c -> (b f) n c")
        
#             return mixed_tokens

        # >>> [ADD-PA-UTIL] -----------------------------------------------
        
        # def _to_tensor(x, device):
        #     """
        #     Robust 轉 tensor：
        #     - 支援 Tensor / ndarray / list of (Tensor or ndarray or list)
        #     - 支援 dict：優先取常見鍵，否則取第一個 value
        #     - 支援 PIL Image
        #     - 盡量把「多幀 list[H,W]」堆疊成 [F,H,W]
        #     """
        #     # 1) torch.Tensor
        #     if isinstance(x, torch.Tensor):
        #         return x.to(device=device)
        
        #     # 2) numpy array / memmap
        #     if isinstance(x, (np.ndarray, np.memmap)):
        #         return torch.from_numpy(np.asarray(x)).to(device=device)
        
        #     # 3) dict：取常見鍵；沒有就取第一個 value
        #     if isinstance(x, dict):
        #         for k in ('tensor', 'arr', 'array', 'data', 'layout', 'mask', 'value'):
        #             if k in x:
        #                 return _to_tensor(x[k], device)
        #         # 退而求其次
        #         return _to_tensor(next(iter(x.values())), device)
        
        #     # 4) list/tuple：嘗試視為「多幀」堆疊
        #     if isinstance(x, (list, tuple)):
        #         if len(x) == 0:
        #             return torch.empty(0, device=device)
        #         # 把每個元素個別轉成 tensor，再嘗試 stack 成 [F, ...]
        #         elems = [_to_tensor(e, device) for e in x]
        #         # 去掉多餘的 batch 維（例如 [1,H,W] -> [H,W]），讓 stack 更穩定
        #         normed = []
        #         for e in elems:
        #             if e.dim() == 3 and e.shape[0] == 1:
        #                 e = e[0]
        #             normed.append(e)
        #         try:
        #             return torch.stack(normed, dim=0)
        #         except Exception:
        #             # 若形狀不齊，嘗試轉成 1D 再 pad（最後手段，基本不會用到）
        #             flat = [e.reshape(-1) for e in normed]
        #             return torch.nn.utils.rnn.pad_sequence(flat, batch_first=True).to(device)
        
        #     # 5) PIL Image
        #     if hasattr(x, "size") and hasattr(x, "mode"):  # 粗略判斷 PIL
        #         import PIL.Image
        #         if isinstance(x, PIL.Image.Image):
        #             arr = np.array(x)
        #             return torch.from_numpy(arr).to(device=device)
        
        #     # 6) fallback
        #     try:
        #         return torch.as_tensor(x, device=device)
        #     except Exception:
        #         arr = np.array(x)
        #         return torch.from_numpy(arr).to(device=device)
        
        # def _as_bool_tensor(x, device):
        #     t = _to_tensor(x, device)
        #     if t.dtype == torch.bool:
        #         return t
        #     # 常見：uint8/float/int -> 非 0 即 True
        #     return (t != 0)
        
        # def _ensure_fhw(t):
        #     """
        #     接受 [F,H,W] / [F,1,H,W] / [F,C,H,W] / [H,W]
        #     統一回傳 [F,H,W]
        #     """
        #     if t.dim() == 2:
        #         # [H,W] 單幀 -> [1,H,W]
        #         t = t.unsqueeze(0)
        #     elif t.dim() == 4:
        #         # [F,C,H,W] -> 取第 0 通道
        #         t = t[:, 0]
        #     assert t.dim() == 3, f"expect [F,H,W], got {t.shape}"
        #     return t
        # # =============================================================================
        
        # def _flatten_fhw(mask_fhw):
        #     return mask_fhw.reshape(-1)
        
        # def _select_Q_key(res_dict, H, W):
        #     """
        #     你的 dict 是用 Q = H*W 當 key（int）。這裡會優先找 Q 精準相等；
        #     若沒有就挑最接近的（避免跑錯層時直接報錯）。
        #     """
        #     Q_target = int(H * W)
        #     keys = list(res_dict.keys())
        #     # 容許 key 可能是字串
        #     keys_int = [int(k) for k in keys]
        #     if Q_target in keys_int:
        #         return Q_target
        #     # 退而求其次：取距離 Q_target 最近的
        #     nearest = min(keys_int, key=lambda q: abs(q - Q_target))
        #     return nearest
        
        # @torch.no_grad()
        # def _build_part_masks_from_resdict(id_masks_by_res, part_masks_by_res, H, W, device):
        #     """
        #     從兩個 dict 取出 [F,H,W] 的：
        #       id_masks_by_res[Q] = {'S': [F,H,W], 'B': [F,H,W]}
        #       part_masks_by_res[Q] = {'S_head': [F,H,W], 'B_head': [F,H,W]}
        #     回傳：
        #       masks_Q = { 'S_head','S_body','B_head','B_body' }  每個 shape=[Q] 的 bool
        #       以及 (F,H,W)
        #     """
        #     # 解析度選 key
        #     q_id   = _select_Q_key(id_masks_by_res,   H, W)
        #     q_part = _select_Q_key(part_masks_by_res, H, W)
        
        #     id_entry   = id_masks_by_res[q_id]
        #     part_entry = part_masks_by_res[q_part]
        
        #     # S_full = _ensure_fhw(_as_bool_tensor(id_entry['S'], device))
        #     # B_full = _ensure_fhw(_as_bool_tensor(id_entry['B'], device))
        #     # S_head = _ensure_fhw(_as_bool_tensor(part_entry['S_head'], device))
        #     # B_head = _ensure_fhw(_as_bool_tensor(part_entry['B_head'], device))
        #     S_full_raw = _to_tensor(id_entry['S'], device)
        #     B_full_raw = _to_tensor(id_entry['B'], device)
        #     S_head_raw = _to_tensor(part_entry['S_head'], device)
        #     B_head_raw = _to_tensor(part_entry['B_head'], device)
            
        #     # 允許非布林型：>0 -> True
        #     S_full = _ensure_fhw((S_full_raw != 0))
        #     B_full = _ensure_fhw((B_full_raw != 0))
        #     S_head = _ensure_fhw((S_head_raw != 0))
        #     B_head = _ensure_fhw((B_head_raw != 0))

        #     # body = full AND (NOT head)
        #     S_body = S_full & (~S_head)
        #     B_body = B_full & (~B_head)
        
        #     F = S_full.shape[0]
        #     masks_Q = {
        #         "S_head": _flatten_fhw(S_head),
        #         "S_body": _flatten_fhw(S_body),
        #         "B_head": _flatten_fhw(B_head),
        #         "B_body": _flatten_fhw(B_body),
        #     }
        #     return masks_Q, (F, H, W)
        
        # def _repeat_for_heads(mask_BQ, heads):
        #     # [B,Q] -> [B*heads, Q]
        #     return mask_BQ.repeat_interleave(heads, dim=0)
        
        # @torch.no_grad()
        # def _compute_key_prototypes_per_group(key_bhqd, masks_BHQ, reduce_dtype=None, eps=1e-6):
        #     BxH, Q, Dh = key_bhqd.shape
        #     if reduce_dtype is None:
        #         reduce_dtype = key_bhqd.dtype
        #     protos = {}
        #     for name, m in masks_BHQ.items():
        #         m = m.to(device=key_bhqd.device)
        #         cnt = m.sum(dim=1).clamp_min(1)                     # [B*H]
        #         w = m.unsqueeze(-1).to(dtype=key_bhqd.dtype)        # [B*H, Q, 1]
        #         s = (key_bhqd * w).sum(dim=1)                       # [B*H, Dh]
        #         proto = (s / cnt.unsqueeze(-1)).to(dtype=reduce_dtype)
        #         protos[name] = proto
        #     return protos
        
        # def _nudge_queries_toward(query_bhqd, target_proto_bhd, row_selector_bhq, alpha=0.6, eps=1e-6):
        #     """
        #     將屬於 row_selector 的那些 Query 朝 target_proto 方向微調：
        #       q <- q + alpha * normalize(target_proto - q_mean_of_rows)
        #     - query_bhqd:       [B*H, Q, Dh]   (可能是 fp16)
        #     - target_proto_bhd: [B*H, Dh]
        #     - row_selector_bhq: [B*H, Q] bool
        #     """
        #     import torch
        #     BxH, Q, Dh = query_bhqd.shape
        #     sel = row_selector_bhq
        #     if not torch.is_tensor(sel) or sel.numel() == 0 or (not sel.any()):
        #         return
        
        #     # 對齊 dtype / device（關鍵修正）
        #     qdtype  = query_bhqd.dtype
        #     qdevice = query_bhqd.device
        #     target_proto_bhd = target_proto_bhd.to(dtype=qdtype, device=qdevice)
        #     sel = sel.to(device=qdevice, dtype=torch.bool)
        #     alpha_t = torch.as_tensor(alpha, dtype=qdtype, device=qdevice)
        
        #     # 被選 rows 的加權平均（用遮罩避免布林索引臨時張量）
        #     w = sel.unsqueeze(-1).to(dtype=qdtype)             # [B*H, Q, 1]
        #     cnt = sel.sum(dim=1).clamp_min(1)                  # [B*H]
        #     q_mean = (query_bhqd * w).sum(dim=1) / cnt.unsqueeze(-1)   # [B*H, Dh]
        
        #     # 方向（保持數值穩定）
        #     direction = torch.nn.functional.normalize(
        #         (target_proto_bhd - q_mean).to(dtype=qdtype), dim=-1, eps=eps
        #     )   # [B*H, Dh]
        
        #     # 構出整個 delta，然後用遮罩加到 query（避免布林索引賦值的 dtype 陷阱）
        #     delta = alpha_t * direction.unsqueeze(1).expand(-1, Q, -1)   # [B*H, Q, Dh]
        #     query_bhqd.add_(w * delta)  # in-place，加速省顯存
        
                # >>> [END ADD-PA-UTIL] -------------------------------------------



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
                
                # enable_part_affinity   = bool(kwargs.get("enable_part_affinity", True))
                # part_affinity_alpha    = float(kwargs.get("part_affinity_alpha", 1.0))
                # part_affinity_repel    = bool(kwargs.get("part_affinity_repel_other", True))
                # part_affinity_beta     = float(kwargs.get("part_affinity_beta", 0.4))
            
            
                # if enable_part_affinity and (id_masks_by_res is not None) and (part_masks_by_res is not None):
                #     device = query.device
                #     print("bias")
                #     # 依目前層的 (h,w) 抽取對應解析度的四類 masks（[Q] bool）
                #     masks_Q, (Fmask, Hmask, Wmask) = _build_part_masks_from_resdict(
                #         id_masks_by_res=id_masks_by_res,
                #         part_masks_by_res=part_masks_by_res,
                #         H=h, W=w, device=device
                #     )
            
                #     # [Q] -> [B,Q] -> [B*H,Q] 與 query/key 的 [B*H, Q, Dh] 對齊
                #     B = int(batch_size // clip_length)
                #     def _expand_b(mQ): return mQ.unsqueeze(0).expand(B, -1)
                #     masks_BQ  = {k: _expand_b(v) for k, v in masks_Q.items()}
                #     masks_BHQ = {k: _repeat_for_heads(v, self.heads) for k, v in masks_BQ.items()}
            
                #     # 為每個群組計算 K 原型向量（[B*H, Dh]）
                #     k_protos = _compute_key_prototypes_per_group(key, masks_BHQ, reduce_dtype=query.dtype)
            
                #     # 1) S_head rows 朝 S_body K 原型微調
                #     _nudge_queries_toward(
                #         query_bhqd=query,
                #         target_proto_bhd=k_protos["S_body"],
                #         row_selector_bhq=masks_BHQ["S_head"],
                #         alpha=part_affinity_alpha
                #     )
            
                #     # 2) B_head rows 朝 B_body K 原型微調
                #     _nudge_queries_toward(
                #         query_bhqd=query,
                #         target_proto_bhd=k_protos["B_body"],
                #         row_selector_bhq=masks_BHQ["B_head"],
                #         alpha=part_affinity_alpha
                #     )
            
                #     # （可選）3) 排斥他人身體
                #     if part_affinity_repel:
                #         _nudge_queries_toward(
                #             query_bhqd=query,
                #             target_proto_bhd=(-k_protos["B_body"]),
                #             row_selector_bhq=masks_BHQ["S_head"],
                #             alpha=part_affinity_beta
                #         )
                #         _nudge_queries_toward(
                #             query_bhqd=query,
                #             target_proto_bhd=(-k_protos["S_body"]),
                #             row_selector_bhq=masks_BHQ["B_head"],
                #             alpha=part_affinity_beta
                #         )
                # print(f"--- key shape is {key.shape}")  # [16, 61440, 40]
                # print(f"--- value shape is {value.shape}") # [16, 61440, 40]
                
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

                        # attention_mask = build_hybrid_mask(
                        #     sequence_length=Q,   # 假設每幀有 hw=4 個 token，總長度 t*hw
                        #     clip_length=clip_length,         # 幀數 t=15 (0..14)
                        #     device=query.device,
                        #     dtype=query.dtype,
                        #     s_clean=clean_A,         # [0..11]
                        #     b_clean=clean_B,         # [0..9]
                        # )
                        
                        
                        # b_tokens = (batch_size)  # = b*f
                        # heads = self.heads
                        # # === [MOD HEAD-PART ONLY-AFTER-CROSS] ===
                        # head_body_bias = _build_head_part_bias_mask(
                        #     h, w, clip_length,
                        #     batch_size_tokens=b_tokens,
                        #     heads=heads,
                        #     device=query.device,
                        #     dtype=query.dtype,
                        #     pull_alpha=kwargs.get("head_pull_alpha", 2.0),
                        #     push_gamma=kwargs.get("head_push_gamma", 0.0),
                        #     cross_start=kwargs.get("CROSS_START", 6),   # <— 只在 t ≥ CROSS_START 生效
                        # )
                        # attention_mask = attention_mask + head_body_bias
                        # # === [END MOD] ===
    
                    hidden_states = _sliced_attention(query, key, value, sequence_length_full_frame, dim, attention_mask, time_causal, h, w, clip_length)
                    encoder_hidden_states = hidden_states

        

                        # print(f"after full frame attn : {encoder_hidden_states.shape}")
                # flow attention can see other 
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

#################################################################################################################
# """
# register the attention controller into the UNet of stable diffusion
# Build a customized attention function `_attention'
# Replace the original attention function with `forward' and `spatial_temporal_forward' in attention_controlled_forward function
# Most of spatial_temporal_forward is directly copy from `video_diffusion/models/attention.py'
# TODO FIXME: merge redundant code with attention.py
# """
# from video_diffusion.prompt_attention.flow_traj_attn import (
#     flow_semantic_traj_attention,
#     normalize_traj_and_mask,
#     reshape_heads_to_batch_dim3
# )

# #from video_diffusion.prompt_attention.semantic_flow_attention_high_efficiency import semantic_flow_fullframe_sreg
# from einops import rearrange
# import torch
# import torch.nn.functional as F
# import math
# from diffusers.utils.import_utils import is_xformers_available
# import numpy as np

# if is_xformers_available():
#     import xformers
#     import xformers.ops
# else:
#     xformers = None


# def register_attention_control(model, controller, text_cond, clip_length, height, width, ddim_inversion):
#     "Connect a model with a controller"
#     def attention_controlled_forward(self, place_in_unet, attention_type='cross'):
#         to_out = self.to_out
#         if type(to_out) is torch.nn.modules.container.ModuleList:
#             to_out = self.to_out[0]
#         else:
#             to_out = self.to_out
        
#         def _attention(query, key, value, is_cross, attention_mask=None):
#             if self.upcast_attention:
#                 query = query.float()
#                 key = key.float()
#             # print("query",query.shape)
#             # print("key",key.shape)
#             attention_scores = torch.baddbmm(
#                 torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
#                 query,
#                 key.transpose(-1, -2),
#                 beta=0,
#                 alpha=self.scale,
#             )
#             #print("attention_scores",attention_scores.shape)
#             if attention_mask is not None:
#                 attention_scores = attention_scores + attention_mask

#             if self.upcast_softmax:
#                 attention_scores = attention_scores.float()

#             # START OF CORE FUNCTION
#             # if not ddim_inversion:
#             attention_probs = controller(reshape_batch_dim_to_temporal_heads(attention_scores), 
#                                         is_cross, place_in_unet)
#             attention_probs = reshape_temporal_heads_to_batch_dim(attention_probs)
#             # END OF CORE FUNCTION

#             attention_probs = attention_probs.softmax(dim=-1)

#             # cast back to the original dtype
#             attention_probs = attention_probs.to(value.dtype)

            
#             # compute attention output
#             hidden_states = torch.bmm(attention_probs, value)

#             # reshape hidden_states
#             hidden_states = reshape_batch_dim_to_heads(hidden_states)
#             return hidden_states

#         def reshape_temporal_heads_to_batch_dim(tensor):
#             head_size = self.heads
#             tensor = rearrange(tensor, " b h s t -> (b h) s t ", h = head_size)
#             return tensor

#         def reshape_batch_dim_to_temporal_heads(tensor):
#             head_size = self.heads
#             tensor = rearrange(tensor, "(b h) s t -> b h s t", h = head_size)
#             return tensor
        
#         def reshape_heads_to_batch_dim3(tensor):
#             batch_size1, batch_size2, seq_len, dim = tensor.shape
#             head_size = self.heads
#             tensor = tensor.reshape(batch_size1, batch_size2, seq_len, head_size, dim // head_size)
#             tensor = tensor.permute(0, 3, 1, 2, 4)
#             return tensor
        
#         def reshape_heads_to_batch_dim(tensor):
#             batch_size, seq_len, dim = tensor.shape
#             head_size = self.heads
#             tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
#             tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
#             return tensor


#         def reshape_batch_dim_to_heads(tensor):
#             batch_size, seq_len, dim = tensor.shape
#             head_size = self.heads
#             tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
#             tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
#             return tensor
        
#         def _memory_efficient_attention_xformers(query, key, value, attention_mask, time_causal=False):
#             # TODO attention_mask
#             query = query.contiguous()
#             key = key.contiguous()
#             value = value.contiguous()
#             ##################################################
#             BxH, Q, Dq = query.shape
#             _,   K, Dk = key.shape
#             assert Dq == Dk, f"Q/K dim mismatch: {Dq} vs {Dk}"

#             attn_bias = None
#             if time_causal:
#                 from xformers.ops import fmha
#                 causal_bias = fmha.attn_bias.LowerTriangularMask()
#                 attn_bias = causal_bias
#             ##################################################
#             hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attn_bias)
#             hidden_states = reshape_batch_dim_to_heads(hidden_states)
#             return hidden_states

#         def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
#             # hidden_states: torch.Size([16, 4096, 320])
#             # encoder_hidden_states: torch.Size([16, 77, 768])
#             # print("========================= Cross Attentoin ===============================")
#             is_cross = encoder_hidden_states is not None
            
#             #encoder_hidden_states = encoder_hidden_states

#             text_cond_frames = text_cond.repeat_interleave(clip_length, 0)     # wrong implementation text_cond.repeat(clip_length,1,1)

#             ######for debug######
#             # text_cond_repeat_interleave = text_cond.repeat_interleave(clip_length, 0)
#             # print("after repeat interleave", text_cond_repeat_interleave.shape, text_cond_repeat_interleave.view(-1)[:20])
#             # text_cond_repeat = text_cond.repeat(clip_length,1,1)
#             # print("First 20 elements after repeat:", text_cond_repeat.shape, text_cond_repeat.view(-1)[:20])
#             ######for debug######

#             encoder_hidden_states = text_cond_frames

#             if self.group_norm is not None:
#                 hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#             query = self.to_q(hidden_states)
#             query = reshape_heads_to_batch_dim(query)

#             if self.added_kv_proj_dim is not None:
#                 key = self.to_k(hidden_states)
#                 value = self.to_v(hidden_states)
#                 encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
#                 encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

#                 key = reshape_heads_to_batch_dim(key)
#                 value = reshape_heads_to_batch_dim(value)
#                 encoder_hidden_states_key_proj = reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
#                 encoder_hidden_states_value_proj = reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

#                 key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
#                 value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
#             else:
#                 encoder_hidden_states = text_cond_frames if encoder_hidden_states is not None else hidden_states
#                 key = self.to_k(encoder_hidden_states)
#                 value = self.to_v(encoder_hidden_states)

#                 key = reshape_heads_to_batch_dim(key)
#                 value = reshape_heads_to_batch_dim(value)

#             if attention_mask is not None:
#                 if attention_mask.shape[-1] != query.shape[1]:
#                     target_length = query.shape[1]
#                     attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
#                     attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

#             if self._use_memory_efficient_attention_xformers and query.shape[-2] > ((height//2) * (width//2)):
#                 # for large attention map of 64X64, use xformers to save memory
#                 # print("xformers")
#                 hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask)
#                 # Some versions of xformers return output in fp32, cast it back to the dtype of the input
#                 hidden_states = hidden_states.to(query.dtype)
#             else:
#                 # print("_attention")
#                 hidden_states = _attention(query, key, value, is_cross=is_cross, attention_mask=attention_mask)
#                 # else:
#                 #     hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

#             # linear proj
#             hidden_states = self.to_out[0](hidden_states)

#             #dropout
#             hidden_states = self.to_out[1](hidden_states)
#             return hidden_states


#         def spatial_temporal_forward(
#             hidden_states,
#             encoder_hidden_states=None,
#             attention_mask=None,
#             clip_length: int = None,
#             SparseCausalAttention_index: list = [-1, 'first']  #list = [0]
#         ):
#             # print("==============Sparse Causal Attention =====================")
#             """
#             Most of spatial_temporal_forward is directly copy from `video_diffusion.models.attention.SparseCausalAttention'
#             We add two modification
#             1. use self defined attention function that is controlled by AttentionControlEdit module
#             2. remove the dropout to reduce randomness
#             FIXME: merge redundant code with attention.py

#             """
#             if (
#                 self.added_kv_proj_dim is not None
#                 or encoder_hidden_states is not None
#                 or attention_mask is not None
#             ):
#                 raise NotImplementedError

#             if self.group_norm is not None:
#                 hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#             query = self.to_q(hidden_states)

#             query = reshape_heads_to_batch_dim(query)


#             key = self.to_k(hidden_states)
#             value = self.to_v(hidden_states)

#             if clip_length is not None:
#                 key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
#                 value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)


#                 #  *********************** Start of Spatial-temporal attention **********
#                 frame_index_list = []
                
#                 if len(SparseCausalAttention_index) > 0:
#                     for index in SparseCausalAttention_index:
#                         if isinstance(index, str):
#                             if index == 'first':
#                                 frame_index = [0] * clip_length
#                             if index == 'last':
#                                 frame_index = [clip_length-1] * clip_length
#                             if (index == 'mid') or (index == 'middle'):
#                                 frame_index = [int((clip_length-1)//2)] * clip_length
#                         else:
#                             assert isinstance(index, int), 'relative index must be int'
#                             frame_index = torch.arange(clip_length) + index
#                             frame_index = frame_index.clip(0, clip_length-1)
                            
#                         frame_index_list.append(frame_index)
#                     # print("frame_index_list",frame_index_list)   [bz, frame, 4096, 320]

#                     key = torch.cat([   key[:, frame_index] for frame_index in frame_index_list   #[bz, frame, 8192, 320])
#                                         ], dim=2)
#                     value = torch.cat([ value[:, frame_index] for frame_index in frame_index_list
#                                         ], dim=2)

                
#                 #  *********************** End of Spatial-temporal attention **********
#                 key = rearrange(key, "b f d c -> (b f) d c", f=clip_length)
#                 value = rearrange(value, "b f d c -> (b f) d c", f=clip_length)
#                 # print("key after rearrange",key.shape)
#                 # print("value after rearrange",value.shape)

#             key = reshape_heads_to_batch_dim(key)
#             value = reshape_heads_to_batch_dim(value)

#             # print("query after head to batch dim",query.shape)
#             # print("key after head to batch dim",key.shape)

#             if torch.isnan(query.reshape(-1)[0]): 
#                 print("nan value query",query.reshape(-1)[:10])
#                 print("nan value key",key.reshape(-1)[:10])
#                 exit()

#             # print("query after reshape heads to batch ",query.shape)
#             # print("key after reshape heads to batch",key.shape)

#             if self._use_memory_efficient_attention_xformers and query.shape[-2] > ((height//2) * (width//2)):
#                 # FIXME there should be only one variable to control whether use xformers
#                 # if self._use_memory_efficient_attention_xformers:
#                 # for large attention map of 64X64, use xformers to save memory
#                 hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask)
#                 # Some versions of xformers return output in fp32, cast it back to the dtype of the input
#                 hidden_states = hidden_states.to(query.dtype)
#             else:
#             # if self._slice_size is None or query.shape[0] // self._slice_size == 1:
#                 hidden_states = _attention(query, key, value, attention_mask=attention_mask, is_cross=False)
#             # else:
#             #     hidden_states = self._sliced_attention(
#             #         query, key, value, hidden_states.shape[1], dim, attention_mask
#             #     )

#             # linear proj
#             hidden_states = self.to_out[0](hidden_states)

#             # dropout
#             hidden_states = self.to_out[1](hidden_states)
#             return hidden_states

#         def build_frame_causal_mask(sequence_length, clip_length, device, dtype,
#                                     include_same_frame=True):
#             """
#             回傳形狀為 (1, Q, K) 的 additive mask，允許位置=0，禁止位置=-inf
#             Q=K=sequence_length=t*hw
#             include_same_frame=True  -> 允許同一幀內互看 (<= 幀下三角)
#             include_same_frame=False -> 僅允許過去幀 (< 幀嚴格下三角)
#             """
#             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#             t = clip_length
#             assert sequence_length % t == 0, "sequence_length 必須能被 clip_length 整除"
#             hw = sequence_length // t
        
#             # 幀層級下三角 (t x t)
#             time_tri = torch.tril(torch.ones(t, t, device=device, dtype=torch.bool),
#                                   diagonal=0 if include_same_frame else -1)  # <= or <
#             # 將每個幀格放大成 hw x hw 的 block（Kronecker 乘積）
#             block = torch.ones(hw, hw, device=device, dtype=torch.bool)
#             frame_block_mask = torch.kron(time_tri, block)  # (t*hw, t*hw) = (Q,K)
        
#             # 轉成 additive mask：允許=0，禁止=-inf
#             finfo = torch.finfo(torch.float32 if dtype == torch.float16 else dtype)
#             additive = torch.where(frame_block_mask, torch.zeros(1, device=device, dtype=torch.float32),
#                                    torch.full((1,), finfo.min, device=device, dtype=torch.float32))
#             # 形狀對齊到 (1, Q, K) 以利 broadcast 到 (B*H, Q, K)
#             additive = additive.view(1, sequence_length, sequence_length)
        
#             # 若你啟用 upcast_softmax，把 mask 也用 float32，避免精度問題
#             return additive


        
#         def _sliced_attention(query, key, value, sequence_length, dim, attention_mask, time_causal):
#             #query (bz*heads, t x h x w, org_dim//heads )
#             # print("================== _sliced attention ====================")
#             is_cross = False
#             batch_size_attention = query.shape[0]   # bz * heads
#             hidden_states = torch.zeros(
#                 (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
#             )

#             slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]

#             if ddim_inversion:
#                 per_frame_len = sequence_length//clip_length
#                 attention_store = torch.zeros((batch_size_attention, clip_length, per_frame_len, per_frame_len), device=query.device, dtype=query.dtype)

#             for i in range(hidden_states.shape[0] // slice_size):
#                 start_idx = i * slice_size
#                 end_idx = (i + 1) * slice_size

#                 query_slice = query[start_idx:end_idx]
#                 key_slice = key[start_idx:end_idx]

#                 if self.upcast_attention:
#                     query_slice = query_slice.float()
#                     key_slice = key_slice.float()

#                 attn_slice = torch.baddbmm(
#                     torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
#                     query_slice,
#                     key_slice.transpose(-1, -2),
#                     beta=0,
#                     alpha=self.scale,
#                 )

#                 if attention_mask is not None:
#                     if time_causal:
#                         attn_slice = attn_slice + attention_mask[:1]
#                     else:
#                         attn_slice = attn_slice + attention_mask[start_idx:end_idx]

#                 if self.upcast_softmax:
#                     attn_slice = attn_slice.float()

#                 if i < self.heads:
#                     if not ddim_inversion:
#                         attention_probs = controller((attn_slice.unsqueeze(1)),is_cross, place_in_unet)
#                         attn_slice = attention_probs.squeeze(1)

#                 attn_slice = attn_slice.softmax(dim=-1)

#                 # cast back to the original dtype
#                 attn_slice = attn_slice.to(value.dtype)
#                 ## bz == 1, sliced head 
#                 if ddim_inversion:
#                     # attn_slice (1, thw, thw)
#                     bz, thw, thw = attn_slice.shape
#                     t = clip_length
#                     hw =  thw // t
#                     # 初始化 per_frame_attention
#                     # (1, t, hxw)

#                     per_frame_attention = torch.empty((bz, t, hw, hw), device=attn_slice.device)

#                     # # 循环提取每一帧的对角线注意力
#                     for idx in range(t):
#                         start_idx_ = idx * hw
#                         end_idx_ = (idx + 1) * hw
#                         # per frame attention extraction
#                         per_frame_attention[:, idx, :, :] = attn_slice[:, start_idx_:end_idx_, start_idx_:end_idx_]

#                         # current_query_block = attn_slice[:, start_idx_:end_idx_, :] 
#                         # aggregated_attention = current_query_block.view(bz, hw, t, hw).mean(dim=2)
#                         # # print('aggregated_attention',aggregated_attention.shape)
#                         # per_frame_attention[:, idx, :, :] = aggregated_attention

#                     per_frame_attention = rearrange(per_frame_attention, "b t h w -> (b t) h w")
#                     attention_store[start_idx:end_idx] = per_frame_attention
                
#                 attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

#                 hidden_states[start_idx:end_idx] = attn_slice
#             if ddim_inversion:
#                 # attention store (bz*heads, t , h, w) h=res, w=res
#                 _ = controller(attention_store, is_cross, place_in_unet)

#             # reshape hidden_states
#             hidden_states = reshape_batch_dim_to_heads(hidden_states)
#             return hidden_states


#         def fully_frame_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, clip_length=None, inter_frame=False, flow_only=True, time_causal=True, **kwargs):
#             print(" ====== attn1 is displayed by attention register (attention_register.py line 377) ======")
#             # print(encoder_hidden_states) # None
            
#             batch_size, sequence_length, _ = hidden_states.shape
#             # print("hidden_states.shape",hidden_states.shape)
#             # print("sequence_length",sequence_length)
#             # print("======================== Full Frame forward ========================")
#             encoder_hidden_states = encoder_hidden_states
#             h = kwargs['height']
#             w = kwargs['width']

#             use_fullframe_layer = False

#             if self.group_norm is not None:
#                 # print("group norm") # no group norm
#                 hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#             query = self.to_q(hidden_states)  # (bf) x d(hw) x c
#             self.q = query
#             if self.inject_q is not None:
#                 # print(f"--- Inject q is {self.inject_q} ---") # NO Inject q
#                 query = self.inject_q
    
#             dim = query.shape[-1]
#             query_old = query.clone()

#             # All frames
#             #init query (bz*t, hxw, dim)
#             # print(f"--- Query shape is {self.q.shape} ---")
#             query = rearrange(query, "(b f) d c -> b (f d) c", f=clip_length)
#             # print(f"--- Rearrange Query shape is {query.shape} ---")
#             query = reshape_heads_to_batch_dim(query)  #(bz*heads, txhxw, dim//heads  [16, 61440, 40]
#             # print(f"--- reshape_heads_to_batch Query shape is {query.shape} ---")
#             if self.added_kv_proj_dim is not None:
#                 raise NotImplementedError
    
#             encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

                
#             key = self.to_k(encoder_hidden_states)
#             self.k = key
#             if self.inject_k is not None:
#                 key = self.inject_k
#             key_old = key.clone()
#             value = self.to_v(encoder_hidden_states)

#             print(f"encoder hidden state : {encoder_hidden_states.shape}") # [30, 4096, 320]
#             value_old = value.clone()
#             print(f"value old : {value_old.shape}") # [30, 4096, 320]

#             # if flow_only and not use_fullframe_layer:
#             if not use_fullframe_layer:
#                 print("323232323232323232323232323")
#                 encoder_hidden_states = rearrange(encoder_hidden_states, "(b f) d c -> b (f d) c", f=clip_length)
#                 hidden_states = encoder_hidden_states

#             # if use_fullframe_layer or (not flow_only):
#             if use_fullframe_layer:
#                 if inter_frame:
#                     print("full is using.")
#                     # print("--- Inter frame is True ---")
#                     key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
#                     value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
#                     key = rearrange(key, "b f d c -> b (f d) c",)
#                     value = rearrange(value, "b f d c -> b (f d) c")
#                 else:
#                     # All frames
#                     print("--- All frame is True ---")
#                     key = rearrange(key, "(b f) d c -> b (f d) c", f=clip_length)
#                     value = rearrange(value, "(b f) d c -> b (f d) c", f=clip_length)
    
#                 key = reshape_heads_to_batch_dim(key)
#                 value = reshape_heads_to_batch_dim(value)
#                 # print(f"--- key shape is {key.shape}")  # [16, 61440, 40]
#                 # print(f"--- value shape is {value.shape}") # [16, 61440, 40]
                
#                 if attention_mask is not None:
#                     if attention_mask.shape[-1] != query.shape[1]:
#                         target_length = query.shape[1]
#                         attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
#                         attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
#                         print(f"--- attention mask shape is {attention_mask.shape}")  # no attention mask
#                 # else:
#                 #     print("--- Attention mask is none. ---")
        
#                 #print("query.shape[0]",query.shape[0])  # 16
#                 self._slice_size = 1   ### 8
#                 sequence_length_full_frame = query.shape[1]
    
#                 # attention, what we cannot get enough of
#                 if self._use_memory_efficient_attention_xformers and query.shape[-2] > clip_length*(32 ** 2):
#                     print("--- use xformer ---")
#                     ############### time_causal = True -> modified full frame attention ###############
#                     hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask, time_causal = time_causal)
    
#                     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
#                     hidden_states = hidden_states.to(query.dtype)
#                     encoder_hidden_states = hidden_states
#                     print(f"after full frame attn : {encoder_hidden_states.shape}")
#                 else:
#                     print("--- slice attention ---")
#                     # if ddim_inversion:
#                     # #if self._slice_size is None or query.shape[0] // self._slice_size == 1:
#                     #     hidden_states = _attention(query, key, value, attention_mask)
#                     # else:
#                     # print(f"attention map is {attention_mask}") #None

#                     if time_causal :
#                         Q = sequence_length_full_frame  # = query.shape[-2]
#                         # 幀級因果遮罩，允許同幀互看（若要嚴格只能看過去幀，把 include_same_frame=False）
#                         attention_mask = build_frame_causal_mask(
#                             sequence_length=Q,
#                             clip_length=clip_length,
#                             device=query.device,
#                             dtype=query.dtype,
#                             include_same_frame=True,
#                         )


#                         hidden_states = _sliced_attention(query, key, value, sequence_length_full_frame, dim, attention_mask, time_causal)
#                         encoder_hidden_states = hidden_states
#                         print(f"after full frame attn : {encoder_hidden_states.shape}")
#                 # flow attention can see other 
#             if controller.__class__.__name__ == "ST_Layout_Attn_ControlEdit" and not (h == 64 and w == 64):
#                 print("flow samentic attention==============")
#                 print(f"--- [h,w] : {[h,w]}, {kwargs['flatten_res']}")
#                 if [h, w] in kwargs['flatten_res']:
#                     print("--- start flow guided attention ---")
            
#                     # -------- Step 1: 前置準備 --------
#                     # (a) 把 encoder_hidden_states 攤回 (B*F, N, C) 供 flow 函式使用
#                     encoder_hidden_states_ff = rearrange(encoder_hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
#                     if self.group_norm is not None:
#                         encoder_hidden_states_ff = self.group_norm(encoder_hidden_states_ff.transpose(1, 2)).transpose(1, 2)
            
#                     # (b) old_qk / flow_only 的 q/k/v 選擇
#                     if kwargs.get("old_qk", 1) == 1:
#                         query_old_ff = query_old
#                         key_old_ff   = key_old
#                     else:
#                         query_old_ff = encoder_hidden_states_ff
#                         key_old_ff   = encoder_hidden_states_ff

#                     # if not flow_only or use_fullframe_layer:
#                     if use_fullframe_layer:
#                         print("ababababababababababababababababababababa")
#                         value_old_ff = encoder_hidden_states
#                     else:
#                         value_old_ff = value_old  # 直接沿用
            
#                     # (c) 正規化 traj/mask -> [F,N,L,*]，並裁到 anchor+過去 clip_length-1
#                     sequence_length = h * w  # N
#                     traj_in = kwargs["traj"]
#                     mask_in = kwargs["mask"]
#                     traj_ff, mask_ff = normalize_traj_and_mask(traj_in, mask_in, F_clip=clip_length, N=sequence_length)
#                     if traj_ff.size(2) >= clip_length:
#                         traj_ff = torch.cat([traj_ff[:, :, 0:1, :], traj_ff[:, :, -clip_length+1:, :]], dim=2)  # [F,N,L,3]
#                         mask_ff = torch.cat([mask_ff[:, :, 0:1],     mask_ff[:, :, -clip_length+1:]],     dim=2)  # [F,N,L]
            
#                     # (d) 準備 _key/_value 供沿軌跡 gather：[(B*F),N,D] -> [B,F,H,W,D]
#                     Bsmall = batch_size // clip_length
#                     _key_ff   = rearrange(key_old_ff,   '(b f) (hh ww) d -> b f hh ww d', b=Bsmall, f=clip_length, hh=h, ww=w)
#                     # value 用「若 flow_only 則 value_old；否則用 encoder_hidden_states_ff」的分支
#                     # value_for_gather = encoder_hidden_states_ff if not flow_only else value_old_ff
#                     value_for_gather = encoder_hidden_states_ff if use_fullframe_layer else value_old_ff
#                     _value_ff = rearrange(value_for_gather, '(b f) (hh ww) d -> b f hh ww d', b=Bsmall, f=clip_length, hh=h, ww=w)
            
#                     # -------- Step 2: 呼叫新 attention --------
#                     # low efficiency
#                     if use_fullframe_layer:
#                         flow_only=False
#                     else:
#                         flow_only=True
                        
#                     hidden_states_ff =  flow_semantic_traj_attention(
#                         query_old=query_old_ff,
#                         key_old=key_old_ff,
#                         value_old=value_old_ff,
#                         encoder_hidden_states=encoder_hidden_states_ff,  # (B*F, N, C)
#                         group_norm=self.group_norm,
#                         traj=traj_ff,                  # [F, N, L, 3]
#                         mask=mask_ff,                  # [F, N, L]
#                         time_causal=time_causal,
#                         _key=_key_ff, _value=_value_ff,   # [B, F, H, W, D]
#                         h=h, w=w,
#                         clip_length=clip_length,
#                         heads=self.heads,
#                         controller=controller,
#                         sem_chunk=getattr(self, "sem_chunk", 64),
#                         use_sem_aug=getattr(self, "use_sem_aug", True),
#                         flow_only=flow_only,
#                         old_qk=kwargs.get("old_qk", 1),
#                         future_lookahead=2,          # 看 1 幀未來
#                         temporal_decay_tau=1.5   
#                     )
    
#                     # -------- Step 3: to_out 投影（與你原本一致） --------
#                     hidden_states_ff = self.to_out[0](hidden_states_ff)
#                     hidden_states_ff = self.to_out[1](hidden_states_ff)
            
#                     # -------- Step 4: 攤回 (B*F, N, C) 並 return --------
#                     hidden_states_ff = rearrange(hidden_states_ff, "b (f d) c -> (b f) d c", f=clip_length)
#                     return hidden_states_ff

#             else:
#             ### original flow attention    
#                 print(f"--- [h,w] : {[h,w]}, {kwargs['flatten_res']}")
#                 if [h,w] in kwargs['flatten_res']:
#                     print("--- start flow guided attention ---")
#                     # if not flow_only:
#                     encoder_hidden_states = rearrange(encoder_hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
#                     if self.group_norm is not None:
#                         encoder_hidden_states = self.group_norm(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
        
#                     if kwargs["old_qk"] == 1:
#                         # print("--- old query and key for attention ---")
#                         query = query_old
#                         key = key_old
#                         # print(f"flow query : {query.shape}") # [30, 4096, 320]
#                         # print(f"flow key : {key.shape}") # [30, 4096, 320]
#                     # else:
#                     #     # print("--- hiden state for attention ---")
#                     #     query = encoder_hidden_states
#                     #     key = encoder_hidden_states
#                     # value = hidden_states
#                     # if not flow_only or use_fullframe_layer:
#                     if use_fullframe_layer:
#                         value = encoder_hidden_states
#                         # print(f"flow value : {value.shape}") # [30, 4096, 320]
#                     else:
#                         # value_old = rearrange(value_old, "b (f d) c -> (b f) d c", f=clip_length)
#                         value = value_old
#                         # print(f"flow old : {value.shape}")
                    
#                     traj = kwargs["traj"]
#                     traj = rearrange(traj, '(f n) l d -> f n l d', f=clip_length, n=sequence_length)
                    
#                     mask = rearrange(kwargs["mask"], '(f n) l -> f n l', f=clip_length, n=sequence_length)
#                     mask = torch.cat([mask[:, :, 0].unsqueeze(-1), mask[:, :, -clip_length+1:]], dim=-1)
#                     # print(f"--- traj shape is {traj.shape} ---")  # traj shape is torch.Size([15, 4096, 39, 3])
#                     # print(f"--- mask shape is {mask.shape} ---")  # mask shape is torch.Size([15, 4096, 15])
#                     #print('traj',traj.shape)
#                     #print('mask',mask.shape)
        
#                     traj_key_sequence_inds = torch.cat([traj[:, :, 0, :].unsqueeze(-2), traj[:, :, -clip_length+1:, :]], dim=-2)
#                     t_inds = traj_key_sequence_inds[:, :, :, 0]
#                     x_inds = traj_key_sequence_inds[:, :, :, 1]
#                     y_inds = traj_key_sequence_inds[:, :, :, 2]
        
#                     ##### attention modification for previous frames #################################
#                     anchor = t_inds[:, :, 0].unsqueeze(-1).expand_as(t_inds)
        
#                     if time_causal:
#                         traj_mask = t_inds <= anchor   
#                     else:
#                         traj_mask = torch.ones_like(t_inds, dtype=torch.bool)
#                     t_inds = torch.where(traj_mask, t_inds, torch.zeros_like(t_inds))
#                     x_inds = torch.where(traj_mask, x_inds, torch.zeros_like(x_inds))
#                     y_inds = torch.where(traj_mask, y_inds, torch.zeros_like(y_inds))
#                     #############################################################
        
                    
#                     # for i in range(14):
#                     #     print(f"--- tinds : {t_inds.shape}, {t_inds[i, 2886]}") # (15, 4096, 15)
#                     #     print(f"--- xinds : {x_inds.shape}, {x_inds[i, 2886]}")
#                     #     print(f"--- yinds : {y_inds.shape}, {y_inds[i, 2886]}")
#                     #     print("-" * 25)
#                     # for j in range(2000, 2020, 1):
#                     #     print(f"--- tinds : {t_inds.shape}, {t_inds[7, j]}") # (15, 4096, 15)
#                     #     print(f"--- xinds : {x_inds.shape}, {x_inds[7, j]}")
#                     #     print(f"--- yinds : {y_inds.shape}, {y_inds[7, j]}")
#                     #     print("+" * 25)
#                     query_tempo = query.unsqueeze(-2)   
#                     print(f"--- query tempo shape: {query_tempo.shape} ---") # (2*15, 4096, 1, 320)
#                     _key = rearrange(key, '(b f) (h w) d -> b f h w d', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
#                     _value = rearrange(value, '(b f) (h w) d -> b f h w d', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
#                     print(f"--- _key shape: {_key.shape} ---") # [2, 15, 64, 64, 320]
#                     print(f"--- _value shape: {_value.shape} ---") # [2, 15, 64, 64, 320]
#                     key_tempo = _key[:, t_inds, x_inds, y_inds] #  [2, 15, 4096, 15, 320])
#                     value_tempo = _value[:, t_inds, x_inds, y_inds] # [2, 15, 4096, 15, 320])
#                     print(f"--- key tempo shape: {key_tempo.shape} ---")
#                     print(f"--- value tempo shape: {value_tempo.shape} ---")
#                     key_tempo = rearrange(key_tempo, 'b f n l d -> (b f) n l d') # [30, 4096, 15, 320]
#                     value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d') # [30, 4096, 15, 320]
#                     print(f"--- key tempo shape: {key_tempo.shape} ---")
#                     print(f"--- value tempo shape: {value_tempo.shape} ---")    
              
#                     ##### attention modification#################################
#                     seq_mask = mask
#                     # print(f"--- original mask shape : {seq_mask.shape}")
#                     # print(f"--- traj_mask shape : {traj_mask.shape}")
#                     keep_mask = seq_mask & traj_mask
#                     mask = rearrange(torch.stack([keep_mask, keep_mask]),  'b f n l -> (b f) n l')
        
#                     #############################################################
#                     # mask = rearrange(torch.stack([mask, mask]),  'b f n l -> (b f) n l')
#                     mask = mask[:,None].repeat(1, self.heads, 1, 1).unsqueeze(-2)
        
                    
#                     attn_bias = torch.zeros_like(mask, dtype=key_tempo.dtype) # regular zeros_like
#                     attn_bias[~mask] = -torch.inf  
        
#                     # print('attn_bias',attn_bias.shape)  (30, H, 1, 4096, 15)
#                     print('query_tempo',query_tempo.shape) # query_tempo torch.Size([30, 4096, 1, 320])
#                     print('key_tempo',key_tempo.shape)     # key_tempo torch.Size([30, 4096, 15, 320])
#                     print('value_tempo',value_tempo.shape) # value_tempo torch.Size([30, 4096, 15, 320])
#                     # flow attention
#                     query_tempo = reshape_heads_to_batch_dim3(query_tempo) # query_tempo torch.Size([30, 8, 4096, 1, 40])
#                     key_tempo = reshape_heads_to_batch_dim3(key_tempo)     # key_tempo torch.Size([30, 8, 4096, 15, 40])
#                     value_tempo = reshape_heads_to_batch_dim3(value_tempo) # value_tempo torch.Size([30, 8, 4096, 15, 40])
#                     # print("-------------------------------------------")
#                     # print('query_tempo',query_tempo.shape)
#                     # print('key_tempo',key_tempo.shape)
#                     # print('value_tempo',value_tempo.shape)
                    
#                     attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(query_tempo.size(-1)) + attn_bias
#                     attn_matrix2 = F.softmax(attn_matrix2, dim=-1)
#                     out = (attn_matrix2@value_tempo).squeeze(-2)
        
#                     hidden_states = rearrange(out,'(b f) k (h w) d -> b (f h w) (k d)', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
        
#                 # linear proj
#                 hidden_states = self.to_out[0](hidden_states)
        
#                 # dropout
#                 hidden_states = self.to_out[1](hidden_states)
        
#                 # All frames
#                 hidden_states = rearrange(hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
#                 return hidden_states


#         if attention_type == 'CrossAttention':
#             # return mod_forward
#             return forward
#         elif attention_type == "SparseCausalAttention":
#             #return mod_forward
#             return spatial_temporal_forward
#         elif attention_type == "FullyFrameAttention":
#             #return mod_forward
#             return fully_frame_forward    

#     class DummyController:

#         def __call__(self, *args):
#             return args[0]

#         def __init__(self):
#             self.num_att_layers = 0

#     if controller is None:
#         controller = DummyController()
    
#     def register_recr(net_, count, place_in_unet):
#         if net_[1].__class__.__name__ == 'CrossAttention' \
#             or net_[1].__class__.__name__ == 'FullyFrameAttention' \
#             or net_[1].__class__.__name__ == 'SparseCausalAttention' :
#             net_[1].forward = attention_controlled_forward(net_[1], place_in_unet, attention_type = net_[1].__class__.__name__)
#             return count + 1
#         elif hasattr(net_[1], 'children'):
#             for net in net_[1].named_children():
#                 if net[0] !='attn_temporal':

#                     count = register_recr(net, count, place_in_unet)

#         return count

#     cross_att_count = 0
#     sub_nets = model.unet.named_children()
#     print(f"===============SubNet================= ")
#     print(sub_nets)
#     print("=======================================")
#     for net in sub_nets:
#         if "down" in net[0]:
#             cross_att_count += register_recr(net, 0, "down")
#         elif "up" in net[0]:
#             cross_att_count += register_recr(net, 0, "up")
#         elif "mid" in net[0]:
#             cross_att_count += register_recr(net, 0, "mid")
#     #print(f"Number of attention layer registered {cross_att_count}")
#     controller.num_att_layers = cross_att_count


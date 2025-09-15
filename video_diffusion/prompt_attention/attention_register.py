"""
register the attention controller into the UNet of stable diffusion
Build a customized attention function `_attention'
Replace the original attention function with `forward' and `spatial_temporal_forward' in attention_controlled_forward function
Most of spatial_temporal_forward is directly copy from `video_diffusion/models/attention.py'
TODO FIXME: merge redundant code with attention.py
"""

from einops import rearrange
import torch
import torch.nn.functional as F
import math
from diffusers.utils.import_utils import is_xformers_available
import numpy as np

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def register_attention_control(model, controller, text_cond, clip_length, height, width, ddim_inversion):
    "Connect a model with a controller"
    def attention_controlled_forward(self, place_in_unet, attention_type='cross'):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
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
        
        def _memory_efficient_attention_xformers(query, key, value, attention_mask, time_causal=False):
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
                hidden_states = _attention(query, key, value, attention_mask=attention_mask, is_cross=False)
            # else:
            #     hidden_states = self._sliced_attention(
            #         query, key, value, hidden_states.shape[1], dim, attention_mask
            #     )

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states
################################################################################################################

        def build_frame_causal_mask(sequence_length, clip_length, device, dtype,
                                    include_same_frame=True):
            """
            回傳形狀為 (1, Q, K) 的 additive mask，允許位置=0，禁止位置=-inf
            Q=K=sequence_length=t*hw
            include_same_frame=True  -> 允許同一幀內互看 (<= 幀下三角)
            include_same_frame=False -> 僅允許過去幀 (< 幀嚴格下三角)
            """
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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

################################################################################################################

        
        def _sliced_attention(query, key, value, sequence_length, dim, attention_mask, time_causal):
            #query (bz*heads, t x h x w, org_dim//heads )
            # print("================== _sliced attention ====================")
            is_cross = False
            batch_size_attention = query.shape[0]   # bz * heads
            hidden_states = torch.zeros(
                (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
            )

            slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]

            if ddim_inversion:
                per_frame_len = sequence_length//clip_length
                attention_store = torch.zeros((batch_size_attention, clip_length, per_frame_len, per_frame_len), device=query.device, dtype=query.dtype)

            for i in range(hidden_states.shape[0] // slice_size):
                start_idx = i * slice_size
                end_idx = (i + 1) * slice_size

                query_slice = query[start_idx:end_idx]
                key_slice = key[start_idx:end_idx]

                if self.upcast_attention:
                    query_slice = query_slice.float()
                    key_slice = key_slice.float()

                attn_slice = torch.baddbmm(
                    torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
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

                if i < self.heads:
                    if not ddim_inversion:
                        attention_probs = controller((attn_slice.unsqueeze(1)),is_cross, place_in_unet)
                        attn_slice = attention_probs.squeeze(1)

                attn_slice = attn_slice.softmax(dim=-1)

                # cast back to the original dtype
                attn_slice = attn_slice.to(value.dtype)
                ## bz == 1, sliced head 
                if ddim_inversion:
                    # attn_slice (1, thw, thw)
                    bz, thw, thw = attn_slice.shape
                    t = clip_length
                    hw =  thw // t
                    # 初始化 per_frame_attention
                    # (1, t, hxw)

                    per_frame_attention = torch.empty((bz, t, hw, hw), device=attn_slice.device)

                    # # 循环提取每一帧的对角线注意力
                    for idx in range(t):
                        start_idx_ = idx * hw
                        end_idx_ = (idx + 1) * hw
                        # per frame attention extraction
                        per_frame_attention[:, idx, :, :] = attn_slice[:, start_idx_:end_idx_, start_idx_:end_idx_]

                        # current_query_block = attn_slice[:, start_idx_:end_idx_, :] 
                        # aggregated_attention = current_query_block.view(bz, hw, t, hw).mean(dim=2)
                        # # print('aggregated_attention',aggregated_attention.shape)
                        # per_frame_attention[:, idx, :, :] = aggregated_attention

                    per_frame_attention = rearrange(per_frame_attention, "b t h w -> (b t) h w")
                    attention_store[start_idx:end_idx] = per_frame_attention
                
                attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

                hidden_states[start_idx:end_idx] = attn_slice
            if ddim_inversion:
                # attention store (bz*heads, t , h, w) h=res, w=res
                _ = controller(attention_store, is_cross, place_in_unet)

            # reshape hidden_states
            hidden_states = reshape_batch_dim_to_heads(hidden_states)
            return hidden_states


        def fully_frame_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, clip_length=None, inter_frame=False, flow_only=False, time_causal=True, **kwargs):
            print(" ====== attn1 is displayed by attention register (attention_register.py line 377) ======")
            # print(encoder_hidden_states) # None
            
            batch_size, sequence_length, _ = hidden_states.shape
            # print("hidden_states.shape",hidden_states.shape)
            # print("sequence_length",sequence_length)
            # print("======================== Full Frame forward ========================")
            encoder_hidden_states = encoder_hidden_states
            h = kwargs['height']
            w = kwargs['width']
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
            if flow_only:
                encoder_hidden_states = rearrange(encoder_hidden_states, "(b f) d c -> b (f d) c", f=clip_length)
                hidden_states = encoder_hidden_states
            print(f"encoder hidden state : {encoder_hidden_states.shape}") # [30, 4096, 320]
            value_old = value.clone()
            print(f"value old : {value_old.shape}") # [30, 4096, 320]
            if not flow_only:

                if inter_frame:
                    print("full is using.")
                    # print("--- Inter frame is True ---")
                    key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
                    value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
                    key = rearrange(key, "b f d c -> b (f d) c",)
                    value = rearrange(value, "b f d c -> b (f d) c")
                else:
                    # All frames
                    print("--- All frame is True ---")
                    key = rearrange(key, "(b f) d c -> b (f d) c", f=clip_length)
                    value = rearrange(value, "(b f) d c -> b (f d) c", f=clip_length)
    
                key = reshape_heads_to_batch_dim(key)
                value = reshape_heads_to_batch_dim(value)
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
                    print("--- use xformer ---")
                    ############### time_causal = True -> modified full frame attention ###############
                    hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask, time_causal = time_causal)
    
                    # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                    hidden_states = hidden_states.to(query.dtype)
                    encoder_hidden_states = hidden_states
                    print(f"after full frame attn : {encoder_hidden_states.shape}")
                else:
                    print("--- slice attention ---")
                    # if ddim_inversion:
                    # #if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                    #     hidden_states = _attention(query, key, value, attention_mask)
                    # else:
                    # print(f"attention map is {attention_mask}") #None
###########################################################################################################
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

###########################################################################################################    
                        hidden_states = _sliced_attention(query, key, value, sequence_length_full_frame, dim, attention_mask, time_causal)
                        encoder_hidden_states = hidden_states
                        print(f"after full frame attn : {encoder_hidden_states.shape}")
                ## flow attention can see other patch
            # if controller.__class__.__name__ == "ST_Layout_Attn_ControlEdit":
            #     print(f"--- [h,w] : {[h,w]}, {kwargs['flatten_res']}")
            #     if [h,w] in kwargs['flatten_res']:
            #         print("--- start flow guided attention ---")
                
            #         encoder_hidden_states = rearrange(encoder_hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
            #         if self.group_norm is not None:
            #             encoder_hidden_states = self.group_norm(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
                
            #         if kwargs["old_qk"] == 1:
            #             query = query_old
            #             key = key_old
            #             print(f"flow query : {query.shape}") # [30, 4096, 320]
            #             print(f"flow key : {key.shape}")     # [30, 4096, 320]
                
            #         if not flow_only:
            #             value = encoder_hidden_states
            #             print(f"flow value : {value.shape}") # [30, 4096, 320]
            #         else:
            #             value = value_old
            #             print(f"flow old : {value.shape}")
                
            #         # ===== traj / seq mask gather =====
            #         traj = kwargs["traj"]
            #         traj = rearrange(traj, '(f n) l d -> f n l d', f=clip_length, n=sequence_length)
                
            #         mask = rearrange(kwargs["mask"], '(f n) l -> f n l', f=clip_length, n=sequence_length)
            #         mask = torch.cat([mask[:, :, 0].unsqueeze(-1), mask[:, :, -clip_length+1:]], dim=-1)
                
            #         traj_key_sequence_inds = torch.cat([traj[:, :, 0, :].unsqueeze(-2), traj[:, :, -clip_length+1:, :]], dim=-2)
            #         t_inds = traj_key_sequence_inds[:, :, :, 0]
            #         x_inds = traj_key_sequence_inds[:, :, :, 1]
            #         y_inds = traj_key_sequence_inds[:, :, :, 2]
                
            #         # ===== 因果遮罩（只能看當前以前）=====
            #         anchor = t_inds[:, :, 0].unsqueeze(-1).expand_as(t_inds)
            #         if time_causal:
            #             traj_mask = t_inds <= anchor
            #         else:
            #             traj_mask = torch.ones_like(t_inds, dtype=torch.bool)
                
            #         t_inds = torch.where(traj_mask, t_inds, torch.zeros_like(t_inds))
            #         x_inds = torch.where(traj_mask, x_inds, torch.zeros_like(x_inds))
            #         y_inds = torch.where(traj_mask, y_inds, torch.zeros_like(y_inds))
                
            #         # ===== reshape K/V 成 (b,f,h,w,d) 再依 flow indices 取樣 =====
            #         _b = int(batch_size/clip_length)
            #         _key   = rearrange(key,   '(b f) (h w) d -> b f h w d', b=_b, f=clip_length, h=h, w=w)
            #         _value = rearrange(value, '(b f) (h w) d -> b f h w d', b=_b, f=clip_length, h=h, w=w)
                
            #         key_tempo   = _key[:,   t_inds, x_inds, y_inds]     # (b,f,n,l,d)
            #         value_tempo = _value[:, t_inds, x_inds, y_inds]     # (b,f,n,l,d)
                
            #         key_tempo   = rearrange(key_tempo,   'b f n l d -> (b f) n l d')
            #         value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d')
                
            #         # ====== A. 硬遮罩：同區塊 ∩ 因果（使用 sreg_map）========================
            #         # 1) 取得 sreg_map (B,Q,Q)，Q=h*w；優先用 self.sreg_maps，否則用 kwargs["sreg_maps"]
            #         sreg_map = controller.sreg_maps[h*w].to(t_inds.device)  
            #         # if hasattr(self, "sreg_maps") and (h*w) in self.sreg_maps:
            #         #     sreg_map = self.sreg_maps[h*w].to(t_inds.device)                  # (b,Q,Q)
            #         # else:
            #         #     assert "sreg_maps" in kwargs, "sreg_maps not found in kwargs"
            #         #     m = kwargs["sreg_maps"]
            #         #     sreg_map = (m[h*w] if isinstance(m, dict) else m).to(t_inds.device)  # (b,Q,Q)
                
            #         # 2) 將 (x_inds,y_inds) 轉為平面索引 q=y*w+x，取 anchor 與候選 q 的相似度
            #         q_tempo  = y_inds * w + x_inds                   # (b,f,n,l)
            #         anchor_q = q_tempo[..., :1]                      # (b,f,n,1)
                
            #         # sreg_map[b, anchor_q, q_tempo] -> (b,f,n,l)
            #         sreg_exp    = sreg_map[:, None, None, None, :, :]     # (b,1,1,1,Q,Q)
            #         anchor_q_exp  = anchor_q.unsqueeze(-1)                  # (b,f,n,1,1)
            #         q_tempo_exp   = q_tempo.unsqueeze(-2)                   # (b,f,n,1,l)
            #         sim_reg = sreg_exp.gather(dim=-2, index=anchor_q_exp.expand(-1, -1, -1, 1, sreg_map.size(-1)))  # (b,f,n,1,Q)
            #         sim_reg = sim_reg.gather(dim=-1, index=q_tempo_exp).squeeze(-2).squeeze(-2)                      # (b,f,n,l)
                
            #         # 3) 同區塊遮罩（沿用 full-frame 的 (mask>0) 判定；如需門檻可改 (sim_reg>=tau)）
            #         same_block_mask = (sim_reg > 0)   # (b,f,n,l) bool
                
            #         # 4) 與既有遮罩交集：seq_mask(原遮罩) ∩ traj_mask(因果) ∩ same_block(同區塊)
            #         seq_mask  = mask
            #         keep_mask = seq_mask & traj_mask & same_block_mask
                
            #         # 5) 形成 attn_bias（被遮掉的 logits = -inf）
            #         mask = rearrange(torch.stack([keep_mask, keep_mask]),  'b f n l -> (b f) n l')
            #         mask = mask[:, None].repeat(1, self.heads, 1, 1).unsqueeze(-2)  # (bf,heads,n,1,l)
                
            #         # ===== build attn & apply =====
            #         attn_bias = torch.zeros_like(mask, dtype=key_tempo.dtype)
            #         attn_bias[~mask] = -torch.inf
                
            #         query_tempo = query.unsqueeze(-2)  # (bf, n, 1, d)
            #         query_tempo = reshape_heads_to_batch_dim3(query_tempo)  # (bf,heads,n,1,d/head)
            #         key_tempo   = reshape_heads_to_batch_dim3(key_tempo)    # (bf,heads,n,l,d/head)
            #         value_tempo = reshape_heads_to_batch_dim3(value_tempo)  # (bf,heads,n,l,d/head)
                
            #         print('query_tempo',query_tempo.shape)
            #         print('key_tempo',key_tempo.shape)
            #         print('value_tempo',value_tempo.shape)
                
            #         attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(query_tempo.size(-1))
            #         attn_matrix2 = attn_matrix2 + attn_bias
            #         attn_matrix2 = F.softmax(attn_matrix2, dim=-1)
            #         out = (attn_matrix2 @ value_tempo).squeeze(-2)
                
            #         hidden_states = rearrange(out,'(b f) k (h w) d -> b (f h w) (k d)', b=_b, f=clip_length, h=h, w=w)
                
            #     # linear proj
            #     hidden_states = self.to_out[0](hidden_states)
            #     hidden_states = self.to_out[1](hidden_states)
            #     hidden_states = rearrange(hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
            #     return hidden_states
            # else:
            ### original flow attention    
            print(f"--- [h,w] : {[h,w]}, {kwargs['flatten_res']}")
            if [h,w] in kwargs['flatten_res']:
                print("--- start flow guided attention ---")
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
                if not flow_only:
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
                key_tempo = rearrange(key_tempo, 'b f n l d -> (b f) n l d') # [30, 64, 64, 320]
                value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d') # [30, 64, 64, 320]
                
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
    
                # print('attn_bias',attn_bias.shape)
                print('query_tempo',query_tempo.shape) # query_tempo torch.Size([30, 4096, 1, 320])
                print('key_tempo',key_tempo.shape)     # key_tempo torch.Size([30, 4096, 15, 320])
                print('value_tempo',value_tempo.shape) # value_tempo torch.Size([30, 4096, 15, 320])
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


        if attention_type == 'CrossAttention':
            # return mod_forward
            return forward
        elif attention_type == "SparseCausalAttention":
            #return mod_forward
            return spatial_temporal_forward
        elif attention_type == "FullyFrameAttention":
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

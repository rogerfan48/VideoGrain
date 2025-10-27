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
    reshape_heads_to_batch_dim3,
    flow_traj_attn_bidir_window,   # <--- 新增
)

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

            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            if self.upcast_softmax:
                attention_scores = attention_scores.float()

            # START OF CORE FUNCTION
            attention_probs = controller(reshape_batch_dim_to_temporal_heads(attention_scores), 
                                        is_cross, place_in_unet)
            attention_probs = reshape_temporal_heads_to_batch_dim(attention_probs)
            # END OF CORE FUNCTION

            attention_probs = attention_probs.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)
            hidden_states = torch.bmm(attention_probs, value)
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
        
        def reshape_heads_to_batch_dim3_local(tensor):
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
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            BxH, Q, Dq = query.shape
            _,   K, Dk = key.shape
            assert Dq == Dk, f"Q/K dim mismatch: {Dq} vs {Dk}"

            attn_bias = None
            if time_causal:
                from xformers.ops import fmha
                causal_bias = fmha.attn_bias.LowerTriangularMask()
                attn_bias = causal_bias

            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attn_bias)
            hidden_states = reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            is_cross = encoder_hidden_states is not None
            text_cond_frames = text_cond.repeat_interleave(clip_length, 0)
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
                hidden_states = hidden_states.to(query.dtype)
            else:
                hidden_states = _attention(query, key, value, is_cross=is_cross, attention_mask=attention_mask)

            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states


        def spatial_temporal_forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            clip_length: int = None,
            SparseCausalAttention_index: list = [-1, 'first']
        ):
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

                    key = torch.cat([key[:, frame_index] for frame_index in frame_index_list], dim=2)
                    value = torch.cat([value[:, frame_index] for frame_index in frame_index_list], dim=2)

                key = rearrange(key, "b f d c -> (b f) d c", f=clip_length)
                value = rearrange(value, "b f d c -> (b f) d c", f=clip_length)

            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            if self._use_memory_efficient_attention_xformers and query.shape[-2] > ((height//2) * (width//2)):
                hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask)
                hidden_states = hidden_states.to(query.dtype)
            else:
                hidden_states = _attention(query, key, value, attention_mask=attention_mask, is_cross=False)

            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        def build_frame_causal_mask(sequence_length, clip_length, device, dtype,
                                    include_same_frame=True):
            t = clip_length
            assert sequence_length % t == 0, "sequence_length must be divisible by clip_length"
            hw = sequence_length // t
            time_tri = torch.tril(torch.ones(t, t, device=device, dtype=torch.bool),
                                  diagonal=0 if include_same_frame else -1)
            block = torch.ones(hw, hw, device=device, dtype=torch.bool)
            frame_block_mask = torch.kron(time_tri, block)
            finfo = torch.finfo(torch.float32 if dtype == torch.float16 else dtype)
            additive = torch.where(frame_block_mask, torch.zeros(1, device=device, dtype=torch.float32),
                                   torch.full((1,), finfo.min, device=device, dtype=torch.float32))
            additive = additive.view(1, sequence_length, sequence_length)
            return additive

        def _sliced_attention(query, key, value, sequence_length, dim, attention_mask, time_causal):
            is_cross = False
            batch_size_attention = query.shape[0]
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
                attn_slice = attn_slice.to(value.dtype)

                if ddim_inversion:
                    bz, thw, thw = attn_slice.shape
                    t = clip_length
                    hw =  thw // t
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

            hidden_states = reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        def fully_frame_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, clip_length=None, inter_frame=False, flow_only=True, time_causal=True, **kwargs):
            batch_size, sequence_length, _ = hidden_states.shape
            h = kwargs['height']
            w = kwargs['width']
            use_fullframe_layer = (h==32 and w==32)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)  # (bf) x d(hw) x c
            if self.inject_q is not None:
                query = self.inject_q
            dim = query.shape[-1]
            query_old = query.clone()

            # All frames → 合併到 (B, F*N, C)
            query = rearrange(query, "(b f) d c -> b (f d) c", f=clip_length)
            query = reshape_heads_to_batch_dim(query)

            if self.added_kv_proj_dim is not None:
                raise NotImplementedError
    
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            if self.inject_k is not None:
                key = self.inject_k
            key_old = key.clone()
            value = self.to_v(encoder_hidden_states)
            value_old = value.clone()

            if not use_fullframe_layer:
                encoder_hidden_states = rearrange(encoder_hidden_states, "(b f) d c -> b (f d) c", f=clip_length)
                hidden_states_ff_full = encoder_hidden_states  # 後面 flow 分支會改用

            if use_fullframe_layer:
                # full-frame attention（slice/xformer）——維持原樣
                key_ff = rearrange(key, "(b f) d c -> b (f d) c", f=clip_length)
                value_ff = rearrange(value, "(b f) d c -> b (f d) c", f=clip_length)
                key_ff = reshape_heads_to_batch_dim(key_ff)
                value_ff = reshape_heads_to_batch_dim(value_ff)

                if attention_mask is not None and attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

                self._slice_size = 1
                sequence_length_full_frame = query.shape[1]

                if self._use_memory_efficient_attention_xformers and query.shape[-2] > clip_length*(32 ** 2):
                    hidden_states = _memory_efficient_attention_xformers(query, key_ff, value_ff, attention_mask, time_causal = time_causal)
                    hidden_states = hidden_states.to(query.dtype)
                    encoder_hidden_states = hidden_states
                else:
                    if time_causal :
                        Q = sequence_length_full_frame
                        attention_mask = build_frame_causal_mask(
                            sequence_length=Q,
                            clip_length=clip_length,
                            device=query.device,
                            dtype=query.dtype,
                            include_same_frame=True,
                        )
                        hidden_states = _sliced_attention(query, key_ff, value_ff, sequence_length_full_frame, dim, attention_mask, time_causal)
                        encoder_hidden_states = hidden_states

            # ========= Flow-guided branch =========
            if controller.__class__.__name__ == "ST_Layout_Attn_ControlEdit" and not (h == 64 and w == 64):
                if [h, w] in kwargs['flatten_res']:
                    # ---------- 準備 (B*F,N,C) / (B,F,H,W,D) ----------
                    encoder_hidden_states_ff = rearrange(encoder_hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
                    if self.group_norm is not None:
                        encoder_hidden_states_ff = self.group_norm(encoder_hidden_states_ff.transpose(1, 2)).transpose(1, 2)

                    if kwargs.get("old_qk", 1) == 1:
                        query_old_ff = query_old
                        key_old_ff   = key_old
                    else:
                        query_old_ff = encoder_hidden_states_ff
                        key_old_ff   = encoder_hidden_states_ff

                    if use_fullframe_layer:
                        value_old_ff = encoder_hidden_states  # 已經 (B, F*N, C)
                    else:
                        value_old_ff = value_old

                    Bsmall = batch_size // clip_length
                    _key_ff   = rearrange(key_old_ff,   '(b f) (hh ww) d -> b f hh ww d', b=Bsmall, f=clip_length, hh=h, ww=w)
                    value_for_gather = encoder_hidden_states_ff if use_fullframe_layer else value_old_ff
                    _value_ff = rearrange(value_for_gather, '(b f) (hh ww) d -> b f hh ww d', b=Bsmall, f=clip_length, hh=h, ww=w)

                    # ---------- Bi-dir window（只看當前與過去） ----------
                    use_bidir = kwargs.get("use_bidir_window", True)
                    if use_bidir:
                        Lwin = int(kwargs.get("bidir_L", 5))
                        alpha = float(kwargs.get("bidir_alpha", 0.5))

                        # 準備 traj/mask（給 bi-dir 內部自己裁窗）
                        sequence_length_hw = h * w
                        traj_in = kwargs["traj"]
                        mask_in = kwargs["mask"]
                        # bidir wrapper 內會自行 normalize / slice，不需先裁
                        per_t_out = []
                        for t in range(clip_length):
                            out_t = flow_traj_attn_bidir_window(
                                query_old=query_old_ff,
                                key_old=key_old_ff,
                                value_old=value_old_ff,
                                encoder_hidden_states=encoder_hidden_states_ff,
                                group_norm=self.group_norm,
                                traj=traj_in, mask=mask_in,
                                _key=_key_ff, _value=_value_ff,
                                h=h, w=w, clip_length=clip_length, heads=self.heads,
                                controller=controller, sem_chunk=getattr(self, "sem_chunk", 64),
                                use_sem_aug=getattr(self, "use_sem_aug", True),
                                flow_only=(not use_fullframe_layer),
                                old_qk=kwargs.get("old_qk", 1),
                                t=t, L=Lwin, fuse_alpha=alpha,
                            )  # [B, N, D]
                            per_t_out.append(out_t)

                        hidden_states_ff = torch.stack(per_t_out, dim=1)         # [B, F, N, D]
                        hidden_states_ff = rearrange(hidden_states_ff, "b f n d -> b (f n) d")
                    else:
                        # 原先一次性 full-history（只 past-causal）
                        # 先正規化為 [F,N,L,*]，並裁到 anchor+過去 clip_length-1
                        sequence_length_hw = h * w
                        traj_in = kwargs["traj"]
                        mask_in = kwargs["mask"]
                        traj_ff, mask_ff = normalize_traj_and_mask(traj_in, mask_in, F_clip=clip_length, N=sequence_length_hw)
                        if traj_ff.size(2) >= clip_length:
                            traj_ff = torch.cat([traj_ff[:, :, 0:1, :], traj_ff[:, :, -clip_length+1:, :]], dim=2)
                            mask_ff = torch.cat([mask_ff[:, :, 0:1],     mask_ff[:, :, -clip_length+1:]],     dim=2)

                        flow_only_flag = not use_fullframe_layer
                        hidden_states_ff = flow_semantic_traj_attention(
                            query_old=query_old_ff,
                            key_old=key_old_ff,
                            value_old=value_old_ff,
                            encoder_hidden_states=encoder_hidden_states_ff,
                            group_norm=self.group_norm,
                            traj=traj_ff, mask=mask_ff, time_causal=time_causal,
                            _key=_key_ff, _value=_value_ff,
                            h=h, w=w, clip_length=clip_length, heads=self.heads,
                            controller=controller,
                            sem_chunk=getattr(self, "sem_chunk", 64),
                            use_sem_aug=getattr(self, "use_sem_aug", True),
                            flow_only=flow_only_flag,
                            old_qk=kwargs.get("old_qk", 1),
                        )

                    # -------- to_out & return --------
                    hidden_states_ff = self.to_out[0](hidden_states_ff)
                    hidden_states_ff = self.to_out[1](hidden_states_ff)
                    hidden_states_ff = rearrange(hidden_states_ff, "b (f d) c -> (b f) d c", f=clip_length)
                    return hidden_states_ff

            # ========= 原本 flow attention（保留） =========
            if [h,w] in kwargs['flatten_res']:
                encoder_hidden_states = rearrange(encoder_hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
                if self.group_norm is not None:
                    encoder_hidden_states = self.group_norm(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)

                if kwargs.get("old_qk", 1) == 1:
                    query_local = query_old
                    key_local   = key_old
                else:
                    query_local = encoder_hidden_states
                    key_local   = encoder_hidden_states

                if use_fullframe_layer:
                    value_local = encoder_hidden_states
                else:
                    value_local = value_old

                traj = kwargs["traj"]
                traj = rearrange(traj, '(f n) l d -> f n l d', f=clip_length, n=h*w)
                mask = rearrange(kwargs["mask"], '(f n) l -> f n l', f=clip_length, n=h*w)
                mask = torch.cat([mask[:, :, 0].unsqueeze(-1), mask[:, :, -clip_length+1:]], dim=-1)

                traj_key_sequence_inds = torch.cat([traj[:, :, 0, :].unsqueeze(-2), traj[:, :, -clip_length+1:, :]], dim=-2)
                t_inds = traj_key_sequence_inds[:, :, :, 0]
                x_inds = traj_key_sequence_inds[:, :, :, 1]
                y_inds = traj_key_sequence_inds[:, :, :, 2]

                anchor = t_inds[:, :, 0].unsqueeze(-1).expand_as(t_inds)
                if time_causal:
                    traj_mask = t_inds <= anchor
                else:
                    traj_mask = torch.ones_like(t_inds, dtype=torch.bool)
                t_inds = torch.where(traj_mask, t_inds, torch.zeros_like(t_inds))
                x_inds = torch.where(traj_mask, x_inds, torch.zeros_like(x_inds))
                y_inds = torch.where(traj_mask, y_inds, torch.zeros_like(y_inds))

                query_tempo = query_local.unsqueeze(-2)
                _key5 = rearrange(key_local, '(b f) (h w) d -> b f h w d', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
                _value5 = rearrange(value_local, '(b f) (h w) d -> b f h w d', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
                key_tempo = _key5[:, t_inds, x_inds, y_inds]
                value_tempo = _value5[:, t_inds, x_inds, y_inds]
                key_tempo = rearrange(key_tempo, 'b f n l d -> (b f) n l d')
                value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d')

                seq_mask = mask
                keep_mask = seq_mask & traj_mask
                mask_bf = rearrange(torch.stack([keep_mask, keep_mask]),  'b f n l -> (b f) n l')
                mask_bf = mask_bf[:,None].repeat(1, self.heads, 1, 1).unsqueeze(-2)

                attn_bias = torch.zeros_like(mask_bf, dtype=key_tempo.dtype)
                attn_bias[~mask_bf] = -torch.inf

                query_tempo = reshape_heads_to_batch_dim3_local(query_tempo)
                key_tempo = reshape_heads_to_batch_dim3_local(key_tempo)
                value_tempo = reshape_heads_to_batch_dim3_local(value_tempo)
                
                attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(query_tempo.size(-1)) + attn_bias
                attn_matrix2 = F.softmax(attn_matrix2, dim=-1)
                out = (attn_matrix2 @ value_tempo).squeeze(-2)

                hidden_states = rearrange(out,'(b f) k (h w) d -> b (f h w) (k d)', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)

            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            hidden_states = rearrange(hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
            return hidden_states


        if attention_type == 'CrossAttention':
            return forward
        elif attention_type == "SparseCausalAttention":
            return spatial_temporal_forward
        elif attention_type == "FullyFrameAttention":
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
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net, 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net, 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net, 0, "mid")
    controller.num_att_layers = cross_att_count

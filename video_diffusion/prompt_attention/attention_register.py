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
        
        def _memory_efficient_attention_xformers(query, key, value, attention_mask):
            # TODO attention_mask
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            # hidden_states: torch.Size([16, 4096, 320])
            # encoder_hidden_states: torch.Size([16, 77, 768])
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
                hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
            
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

        def _sliced_attention(query, key, value, sequence_length, dim, attention_mask):
            #query (bz*heads, t x h x w, org_dim//heads )

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


        def fully_frame_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, clip_length=None, inter_frame=False, **kwargs):
            batch_size, sequence_length, _ = hidden_states.shape
            # print("hidden_states.shape",hidden_states.shape)
            # print("sequence_length",sequence_length)

            encoder_hidden_states = encoder_hidden_states
            h = kwargs['height']
            w = kwargs['width']
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)  # (bf) x d(hw) x c
            self.q = query
            if self.inject_q is not None:
                query = self.inject_q
            dim = query.shape[-1]
            query_old = query.clone()

            # All frames
            #init query (bz*t, hxw, dim)
            query = rearrange(query, "(b f) d c -> b (f d) c", f=clip_length)
            query = reshape_heads_to_batch_dim(query)  #(bz*heads, txhxw, dim//heads)

            if self.added_kv_proj_dim is not None:
                raise NotImplementedError

            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            self.k = key
            if self.inject_k is not None:
                key = self.inject_k
            key_old = key.clone()
            value = self.to_v(encoder_hidden_states)

            if inter_frame:
                key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
                value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)[:, [0, -1]]
                key = rearrange(key, "b f d c -> b (f d) c",)
                value = rearrange(value, "b f d c -> b (f d) c")
            else:
                # All frames
                key = rearrange(key, "(b f) d c -> b (f d) c", f=clip_length)
                value = rearrange(value, "(b f) d c -> b (f d) c", f=clip_length)

            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            #print("query.shape[0]",query.shape[0])  # 16
            self._slice_size = 1   ### 8
            sequence_length_full_frame = query.shape[1]

            # attention, what we cannot get enough of
            if self._use_memory_efficient_attention_xformers and query.shape[-2] > clip_length*(32 ** 2):
                hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
                
            else:
                # if ddim_inversion:
                # #if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                #     hidden_states = _attention(query, key, value, attention_mask)
                # else:
                hidden_states = _sliced_attention(query, key, value, sequence_length_full_frame, dim, attention_mask)

            if [h,w] in kwargs['flatten_res']:
                hidden_states = rearrange(hidden_states, "b (f d) c -> (b f) d c", f=clip_length)
                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                if kwargs["old_qk"] == 1:
                    query = query_old
                    key = key_old
                else:
                    query = hidden_states
                    key = hidden_states
                value = hidden_states

                traj = kwargs["traj"]
                traj = rearrange(traj, '(f n) l d -> f n l d', f=clip_length, n=sequence_length)
                mask = rearrange(kwargs["mask"], '(f n) l -> f n l', f=clip_length, n=sequence_length)
                mask = torch.cat([mask[:, :, 0].unsqueeze(-1), mask[:, :, -clip_length+1:]], dim=-1)

                #print('traj',traj.shape)
                #print('mask',mask.shape)

                traj_key_sequence_inds = torch.cat([traj[:, :, 0, :].unsqueeze(-2), traj[:, :, -clip_length+1:, :]], dim=-2)
                t_inds = traj_key_sequence_inds[:, :, :, 0]
                x_inds = traj_key_sequence_inds[:, :, :, 1]
                y_inds = traj_key_sequence_inds[:, :, :, 2]

                query_tempo = query.unsqueeze(-2)
                _key = rearrange(key, '(b f) (h w) d -> b f h w d', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
                _value = rearrange(value, '(b f) (h w) d -> b f h w d', b=int(batch_size/clip_length), f=clip_length, h=h, w=w)
                key_tempo = _key[:, t_inds, x_inds, y_inds]
                value_tempo = _value[:, t_inds, x_inds, y_inds]
                key_tempo = rearrange(key_tempo, 'b f n l d -> (b f) n l d')
                value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d')

                mask = rearrange(torch.stack([mask, mask]),  'b f n l -> (b f) n l')
                mask = mask[:,None].repeat(1, self.heads, 1, 1).unsqueeze(-2)
                attn_bias = torch.zeros_like(mask, dtype=key_tempo.dtype) # regular zeros_like
                attn_bias[~mask] = -torch.inf

                # print('attn_bias',attn_bias.shape)
                # print('query_tempo',query_tempo.shape)
                # print('key_tempo',key_tempo.shape)

                # flow attention
                query_tempo = reshape_heads_to_batch_dim3(query_tempo)
                key_tempo = reshape_heads_to_batch_dim3(key_tempo)
                value_tempo = reshape_heads_to_batch_dim3(value_tempo)

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
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net, 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net, 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net, 0, "mid")
    #print(f"Number of attention layer registered {cross_att_count}")
    controller.num_att_layers = cross_att_count

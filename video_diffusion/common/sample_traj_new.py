def sample_trajectories_new(video_path, device,height,width):
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large
    print("~~~~~~~~~~~~ sample new trajectory ~~~~~~~~~~~~~~~~")
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
    print(f"--- frames length : {len(frames)} ---")
    clips = list(range(len(frames)))
    
    #=============== raft-large estimate forward optical flow============#
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()
    finished_trajectories = []

    current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], transforms, height,width)
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    print(f"--- optical flow iterate for {len(list_of_flows)} times ---")
    predicted_flows = list_of_flows[-1]
    print(f"predicte_flow shape is : {predicted_flows.shape}") # [14, 2, 512, 512])
    #=============== raft-large estimate forward optical flow============#

    predicted_flows = predicted_flows/max(height,width)

    resolutions =[(height//8,width//8),(height//16,width//16),(height//32,width//32),(height//64,width//64)]
    #resolutions = [64, 32, 16, 8]
    res = {}
    window_sizes = {(height//8,width//8): 2,
                    (height//16,width//16): 1,
                    (height//32,width//32): 1,
                    (height//64,width//64): 1}
    s
    for resolution in resolutions:
        print("="*30)
        print(resolution)
        print('window_sizes[resolution]',window_sizes[resolution])
        trajectories = {}
        height_scale_factor = resolution[0] / height
        width_scale_factor = resolution[1] / width
        predicted_flow_resolu = torch.round(max(resolution[0], resolution[1])*torch.nn.functional.interpolate(predicted_flows, scale_factor=(height_scale_factor, width_scale_factor)))
        print(f"--- predicted flow resolution shape (T, xy, H, W) : {predicted_flow_resolu.shape}")
        T = predicted_flow_resolu.shape[0]+1
        H = predicted_flow_resolu.shape[2]
        W = predicted_flow_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            flow = predicted_flow_resolu[t]  # (2, H, W)  dx and dy
            for h in range(H):
                for w in range(W):

                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        # this point has not been traversed, start new trajectory
                        x = h + int(flow[1, h, w])
                        y = w + int(flow[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            # trajectories.append([(t, h, w), (t+1, x, y)])
                            trajectories[(t, h, w)]= (t+1, x, y)    # key : (t, h, w) 當前位置 ， value : (t+1, x, y) 下一幀 (ㄓㄥˋ )位置
        print(f"--- trajectories length : {len(trajectories)}")
        print(f"--- trajectories ten keys : {list(trajectories.keys())[:10]}")
        conflict_points = keys_with_same_value(trajectories)
        for k in conflict_points:
            index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
            conflict_points[k].pop(index_to_pop)
            for point in conflict_points[k]:
                if point[0] != T-1:
                    trajectories[point]= (-1, -1, -1) # stupid padding with (-1, -1, -1)

        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x//H, x%H):0 for x in range(H*W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj)
            active_traj = new_active_traj
            active_traj+=[[pixel] for pixel in pixel_set if pixel_set[pixel] == 0]
        all_traj += active_traj # [[(0,0,0), (0,1,0)...(-1, -1, -1)], [(0,2,0), (1,1,0)...]]
        
        useful_traj = [i for i in all_traj if len(i)>1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1):
                useful_traj[idx] = useful_traj[idx][:-1]
        print(f"--- num trajectories : {len(useful_traj)} ---")
        print(f"--- longest length  : {max(len(t) for t in useful_traj) if useful_traj else 0} ---")
        print(f"--- total points    : {sum(len(t) for t in useful_traj)}")
        for i, traj in enumerate(useful_traj[:3]):
            print(f"[traj {i}] len={len(traj)}  head={traj[:5]}")

        print("how many points in all trajectories for resolution{}?".format(resolution), sum([len(i) for i in useful_traj]))
        print("how many points in the video for resolution{}?".format(resolution), T*H*W)

        # validate if there are no duplicates in the trajectories
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        print(f"--- trajs shape : {len(trajs)} ---")
        assert len(find_duplicates(trajs)) == 0, "There should not be duplicates in the useful trajectories."

        # check if non-appearing points + appearing points = all the points in the video
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        left_points = all_points- set(trajs)
        print("How many points not in the trajectories for resolution{}?".format(resolution), len(left_points))
        for p in list(left_points):
            useful_traj.append([p])
        print("how many points in all trajectories for resolution{} after pending?".format(resolution), sum([len(i) for i in useful_traj]))


        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1
        print(f"--- longest length : {longest_length} ---")
        print(f"--- sequence length : {sequence_length} ---")
        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:   # p : (0, 1, 1)
                point_to_traj[p] = traj # value : (0, 1, 1), key : traj 

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)
                    sequence = [(t,x,y)]+neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t,x,y)].copy()
                    traj.remove((t,x,y))
                    sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res["traj{}".format(resolution[0])] = seqs
        res["mask{}".format(resolution[0])] = masks
    return res


       def _cos(a, b, eps=1e-8):
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            if na < eps or nb < eps:
                return -math.inf
            return float(np.dot(a, b) / (na * nb))
            
        def _agg_vec(vecs, mode, ema_beta=0.6):
            """
            vecs: list of np.array([dx,dy], float)
            mode: "mean" | "median" | "ema"
            """
            import numpy as np
            if len(vecs) == 0:
                return None
            if mode == "mean":
                return np.mean(np.stack(vecs, 0), axis=0)
            elif mode == "median":
                return np.median(np.stack(vecs, 0), axis=0)
            elif mode == "ema":
                v = vecs[0].astype(float)
                beta = float(ema_beta)
                for i in range(1, len(vecs)):
                    v = beta * v + (1.0 - beta) * vecs[i]
                return v
            else:
                # 預設 mean
                return np.mean(np.stack(vecs, 0), axis=0) 
                
        def _collect_forward_vecs_from_traj(start_key, steps, trajectories):
            """
            從某個離散像素點開始，沿著 trajectories 最多走 `steps` 次，
            用「離散座標差」堆疊向量（每一步：下一點 - 當前點）。
            回傳: list[np.array([dy, dx])]
            """
            import numpy as np
            vecs = []
            cur = start_key  # (t,x,y)
            for _ in range(steps):
                nxt = trajectories.get(cur, None)
                if not nxt or nxt == (-1,-1,-1): 
                    break
                # 離散位移（注意你現有定義：向量是 [Δy, Δx]）
                dy = nxt[2] - cur[2]
                dx = nxt[1] - cur[1]
                vecs.append(np.array([dy, dx], float))
                cur = nxt
            return vecs
            
        def _build_reverse_index(trajectories):
            """
            建立反向索引： (t+1,x,y) -> (t,h,w)
            只收有效連結（不含 -1,-1,-1）
            """
            rev = {}
            for k, v in trajectories.items():
                if v != (-1, -1, -1):
                    rev[v] = k
            return rev

        def _collect_backward_vecs_from_traj(start_key, steps, reverse_index):
            """
            從 (t,h,w) 往過去最多 steps 步，回傳 [dy,dx] 列表
            """
            import numpy as np
            vecs = []
            cur = start_key  # (t,h,w)
            for _ in range(steps):
                if cur not in reverse_index:
                    break
                prev = reverse_index[cur]      # (t-1, h0, w0)
                dy = cur[2] - prev[2]
                dx = cur[1] - prev[1]
                vecs.append(np.array([dy, dx], float))
                cur = prev
            return vecs

        def complict_track_source_flow(
            trajectories,         # dict: {(t,h,w) -> (t+1,x,y) 或 (-1,-1,-1)}
            conflict_points,      # dict: {(t+1,x,y): [(t,h1,w1), (t,h2,w2), ...]}
            T,                    # 總幀數
            # flow=None,            # (可選) 浮點光流 (T-1,2,H,W)
            k_tgt=3,              # 目標往未來看的步數
            k_src=2,              # 來源往過去看的步數
            agg="ema",            # "ema" | "mean" | "median"
            ema_beta=0.6,         # ema 係數
            w_dir=1.0,            # 方向一致性權重
            w_mag=0.5,            # 幅度一致性權重
            w_pos=0.5,            # 位置一致性權重
            sigma_mag=1.0,        # 幅度一致的溫度
            sigma_pos=1.0,        # 位置一致的σ（像素）
            debug=True
        ):
            """
            決策規則：
              - 目標向量：優先用軌跡 (t+1 → t+2 → ...) 的多步 [dy,dx] 聚合；沒有就用 flow[t+1] 後備
              - 來源向量：優先用反向索引回溯 (t0 ← t0-1 ← ...) 的多步 [dy,dx] 聚合；沒有就用 flow[t0-1] 後備
              - 打分：w_dir*cos + w_mag*exp(-|Δmag|/σm) + w_pos*exp(-d^2/(2σp^2))
              - 最後保留最佳來源，其餘來源標記為 (-1,-1,-1)
            """
            import numpy as np, math
        
            reverse_index = _build_reverse_index(trajectories)
        
            def best_target_vec(k):
                tk, x, y = k
                vecs = _collect_forward_vecs_from_traj(k, steps=k_tgt, trajectories=trajectories)
                # if len(vecs) == 0:
                #     # 後備：用 t+1 的 forward flow
                #     if 0 <= tk < T-1:
                #         fv = _flow_vec_forward(flow, tk, x, y)
                #         if fv is not None:
                #             vecs = [np.array(fv, float)]
                return _agg_vec(vecs, mode=agg, ema_beta=ema_beta) if len(vecs) > 0 else None
        
            def best_source_vec(src):
                t0, h, w = src
                vecs = _collect_backward_vecs_from_traj(src, steps=k_src, reverse_index=reverse_index)
                # if len(vecs) == 0:
                #     # 後備：用 (t0-1 -> t0) 的 forward flow
                #     fb = _flow_vec_backward(flow, t0, h, w)
                #     if fb is not None:
                #         vecs = [np.array(fb, float)]
                return _agg_vec(vecs, mode=agg, ema_beta=ema_beta) if len(vecs) > 0 else None
        
            def score(src_vec, tgt_vec, src_pos, tgt_pos):
                # 方向
                cosv = _cos(src_vec, tgt_vec) 
                # if tgt_vec is not None else -1.0
                # 幅度一致
                # m_src = float(np.linalg.norm(src_vec))
                # m_tgt = float(np.linalg.norm(tgt_vec)) if tgt_vec is not None else 0.0
                # mag_pen = math.exp(-abs(m_src - m_tgt) / float(sigma_mag))
        
                # 位置一致（用離散一步近似：src_pos + round(src_vec) ≈ tgt_pos）
                # dy, dx = src_vec
                # pred_x = src_pos[0] + int(round(dx))
                # pred_y = src_pos[1] + int(round(dy))
                # pos_err = math.sqrt((pred_x - tgt_pos[0])**2 + (pred_y - tgt_pos[1])**2)
                # pos_pen = math.exp(-(pos_err**2) / (2.0 * (float(sigma_pos)**2)))
                return cosv
                # return w_dir * cosv + w_mag * mag_pen + w_pos * pos_pen, (cosv, mag_pen, pos_pen)
        
            for k, src_list in conflict_points.items():
                tgt_vec = best_target_vec(k)
                if debug:
                    print(f"--- tgt_pt : {k} comes from {src_list}.")
                if tgt_vec is None:
                    # 沒有 target_vec：退回幅度最大（用離散一步）
                    best_src, best_mag = None, -math.inf
                    for (t0, h, w) in src_list:
                        v = np.array([k[2]-w, k[1]-h], float)  # [dy,dx]
                        mag = float(np.linalg.norm(v))
                        if mag > best_mag:
                            best_mag, best_src = mag, (t0, h, w)
                    for (t0, h, w) in src_list:
                        if (t0, h, w) != best_src and t0 != T-1:
                            trajectories[(t0, h, w)] = (-1, -1, -1)
                    if debug:
                        print(f"--- [no tgt_vec] keep {best_src}, drop others for target {k}")
                    continue
        
                # 正常路徑：做綜合打分
                best_src, best_score = None, -math.inf
                if debug:
                    print(f"--- [target {k}] tgt_vec={tgt_vec}")
        
                for (t0, h, w) in src_list:
                    s_vec = best_source_vec((t0, h, w))
                    
                    # if s_vec is None:
                        
                    #     s = -1e9; parts = (-1, 0, 0)
                    # else:
                    if s_vec is None:
                        # 後備：用一步的離散向量（來源→目標）
                        print(f"--- [first frame] no s_vec")
                        s_vec = np.array([k[2]-w, k[1]-h], float)
                    # s, parts = score(s_vec, tgt_vec, src_pos=(h, w), tgt_pos=(k[1], k[2]))
                    s = score(s_vec, tgt_vec, src_pos=(h, w), tgt_pos=(k[1], k[2]))
                    if debug:
                        print(f"s_vec={s_vec}, tgt_vec={tgt_vec}, score={s:.4f}")
                    if s > best_score:
                        best_score = s
                        best_src = (t0, h, w)
        
                # 截斷其他來源
                for (t0, h, w) in src_list:
                    if (t0, h, w) != best_src and t0 != T-1:
                        trajectories[(t0, h, w)] = (-1, -1, -1)
                if debug:
                    print(f" -> keep {best_src} for target {k}, score={best_score:.4f}")
                print("=" * 100)
        
            return trajectories

     # ##############easy cosine similarity to decide source flow #############################################################
     #    def _cos(a, b, eps=1e-8):
     #        na = np.linalg.norm(a); nb = np.linalg.norm(b)
     #        if na < eps or nb < eps:
     #            return -math.inf
     #        return float(np.dot(a, b) / (na * nb))
                    
     #    def easy_track_source_flow(
     #        trajectories,        # dict: {(t,h,w) -> (t+1,x,y) 或 (-1,-1,-1)}
     #        conflict_points,     # dict: {(t+1,x,y): [(t,h1,w1), (t,h2,w2), ...]}
     #        T,                   # 總幀數
     #    ):
     #        """
     #        用 (t+1,x,y) → (t+2,x2,y2) 的 '未來方向' 當基準，挑選來源。
     #        若 (t+1,x,y) 沒有下一跳，且提供了 flow_{t+1}，就用該點的 (dx,dy) 當基準。
     #        """
     #        import numpy as np
     #        import math
     #        for k, src_list in conflict_points.items():
     #            # k = (t+1, x, y)
     #            tk, x, y = k
     #            # assert tk == t + 1, "conflict key 時間步應為 t+1"
        
     #            # 1) 直接從 trajectories 取 (t+2,x2,y2) 算 target_vec
     #            target_vec = None
     #            if (tk < T-1) and (k in trajectories):
     #                nxt = trajectories[k]  # (t+2, x2, y2) 或 (-1,-1,-1)
     #                if nxt != (-1, -1, -1) and nxt[0] == tk + 1:
     #                    _, x2, y2 = nxt
     #                    target_vec = np.array([y2 - y, x2 - x], dtype=float)
        
     #            # # 2) 若沒有下一跳，且有 flow_{t+1}，用 (dx,dy)
     #            # if target_vec is None and (flow_np is not None):
     #            #     dx = float(flow_np[0, x, y])  # 注意：你先前的慣例 flow[0]=dx, flow[1]=dy
     #            #     dy = float(flow_np[1, x, y])
     #            #     target_vec = np.array([dx, dy], dtype=float)
        
     #            # # 3) 若仍然沒有基準（最後一幀或取不到），那這個衝突就跳過或退回備選規則
     #            # if target_vec is None:
     #            #     # 可選：退回用幅度決勝
     #            #     best_src = None
     #            #     best_mag = -math.inf
     #            #     for (t0, h, w) in src_list:
     #            #         v = np.array([y - w, x - h], dtype=float)
     #            #         mag = float(np.linalg.norm(v))
     #            #         if mag > best_mag:
     #            #             best_mag = mag
     #            #             best_src = (t0, h, w)
     #            #     # 截斷其他
     #            #     for (t0, h, w) in src_list:
     #            #         if (t0, h, w) != best_src and t0 != T-1:
     #            #             trajectories[(t0, h, w)] = (-1, -1, -1)
     #            #     continue
        
     #            # 基於 target_vec 比 cosine
     #            best_src = None
     #            best_cos = -math.inf
     #            best_mag = -math.inf
        
     #            for (t0, h, w) in src_list:
     #                src_vec = np.array([y - w, x - h], dtype=float)  # source -> target
     #                cosv = _cos(src_vec, target_vec)
     #                mag  = float(np.linalg.norm(src_vec))
     #                if (cosv > best_cos) or (cosv == best_cos and mag > best_mag):
     #                    best_cos = cosv
     #                    best_mag = mag
     #                    best_src = (t0, h, w)
        
     #            # 截斷其他來源
     #            for (t0, h, w) in src_list:
     #                if (t0, h, w) != best_src and t0 != T-1:
     #                    trajectories[(t0, h, w)] = (-1, -1, -1)
        
     #        return trajectories
                    
     #    trajectories = easy_track_source_flow(trajectories, conflict_points, T)
        
        
        
        
                
     #    ########################################################################################################################
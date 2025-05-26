Reward_Dic = {
"multicomp/SumoHumans-v0_0": 
"""class RewardCalculator:
    def __init__(self):
        self._stats = {"wins": 0, "games": 0}

    def compute(self, s1, s2, a2, victory_status, s_o, reward_adv, reward_opp):
        # ---------- win-rate tracker ----------
        # if not hasattr(reward, "_stats"):
        #     reward._stats = {"wins": 0, "games": 0}
        if victory_status in ("win", "loss"):
            self._stats["games"] += 1
            self._stats["wins"] += (victory_status == "win")
        win_rate = self._stats["wins"] / max(self._stats["games"], 1)

        # ---------- cast ----------
        s1, s2, s_o, a2 = map(lambda x: np.asarray(x, dtype=np.float32), (s1, s2, s_o, a2))
        arena_r     = s2[391] if s2.size > 391 else 3.0
        remain_time = s2[394] if s2.size > 394 else 0.0

        # ---------- standing score ----------
        torso_h   = np.tanh(s2[2] - 1.0)
        torso_up  = s2[390] if s2.size > 390 else 1.0
        torso_vel = -0.5 * np.tanh(np.linalg.norm(s2[24:27]))
        stand_s   = 0.5*torso_h + 0.4*torso_up + 0.1*torso_vel         # (-1,1)

        # ---------- combat score (细粒度) ----------
        d_prev    = np.linalg.norm(s1[380:382])
        d_curr    = np.linalg.norm(s2[380:382])
        approach  = np.tanh((d_prev - d_curr) / (arena_r + 1e-6))

        self_edge = s2[392] if s2.size > 392 else arena_r
        opp_edge  = s2[393] if s2.size > 393 else arena_r
        ring      = np.tanh((opp_edge - self_edge) / (arena_r + 1e-6))

        opp_up_p  = s1[390] if s1.size > 390 else 1.0
        opp_up_c  = s_o[390] if s_o.size > 390 else 1.0
        tilt_ang  = np.clip(opp_up_p - opp_up_c, 0.0, 1.0)             # 0-1
        tilt_s    = tilt_ang ** 0.5                                    # γ=0.5

        opp_height = s_o[2] if s_o.size > 2 else 2.0
        push_down  = np.clip(1.0 - opp_height, 0.0, 1.0)               # 连续深度

        aggression = 1.0 if d_curr < 0.6 else 0.0

        combat_raw = (0.25*approach + 0.25*ring +
                    0.6 *tilt_s  + 1.0*push_down +
                    0.1 *aggression)
        combat_s   = np.sign(combat_raw) * min(abs(combat_raw), 6.0)   # clip ±6
        combat_s   = combat_s / 6.0                                    # 归一化到 (-1,1)

        # ---------- adaptive weights ----------
        stand_w  = max(0.15, 0.6 * np.exp(-3.0 * win_rate))
        combat_w = 1.0 - stand_w
        dense_r  = stand_w*stand_s + combat_w*combat_s                 # (-1,1)

        # ---------- penalties ----------
        energy_pen = -0.0012 * float(np.sum(a2 ** 2))
        step_pen   = -0.002

        # ---------- sparse / terminal ----------
        sparse_r    = 1.2 * reward_adv - 0.2 * reward_opp
        victory_sig = {"win": 40.0, "loss": -40.0}.get(victory_status, 0.0)
        early_bonus = 0.1 * remain_time if victory_status == "win" else 0.0

        # ---------- total ----------
        total = dense_r + energy_pen + step_pen + sparse_r + victory_sig + early_bonus

        info = {
            "total_reward":   float(total),
            "dense_reward":   float(dense_r),
            "standing_score": float(stand_s),
            "combat_score":   float(combat_s),
            "approach":       float(approach),
            "ring":           float(ring),
            "tilt_s":         float(tilt_s),
            "push_down":      float(push_down),
            "aggression":     float(aggression),
            "stand_weight":   float(stand_w),
            "combat_weight":  float(combat_w),
            "energy_penalty": float(energy_pen),
            "step_penalty":   float(step_pen),
            "sparse_reward":  float(sparse_r),
            "victory_signal": float(victory_sig),
            "early_bonus":    float(early_bonus),
            "win_rate":       float(win_rate),
            "winning_rate":   float(win_rate),
        }
        return float(total), info

    """,

"multicomp/SumoHumans-v0_1": 
"""class RewardCalculator:
    def __init__(self):
        self._stats = {"wins": 0, "games": 0}

    def compute(self, s1, s2, a2, victory_status, s_o, reward_adv, reward_opp):
        # -------- persistent stats (胜率记录) --------
        # if not hasattr(reward, "_stats"):
        #     reward._stats = {"wins": 0, "games": 0}
        if victory_status in ("win", "loss"):
            self._stats["games"] += 1
            self._stats["wins"] += (victory_status == "win")
        winning_rate = self._stats["wins"] / max(self._stats["games"], 1)
        winning_rate = 0.3

        # -------- numpy cast --------
        s1, s2, s_o, a2 = map(lambda x: np.asarray(x, dtype=np.float32), (s1, s2, s_o, a2))

        # -------- arena & time --------
        arena_r     = s2[391] if s2.size > 391 else 3.0
        self_edge   = s2[392] if s2.size > 392 else arena_r
        opp_edge    = s2[393] if s2.size > 393 else arena_r
        remain_time = s2[394] if s2.size > 394 else 0.0
        time_frac   = remain_time / 400.0

        # -------- standing score --------
        height_norm  = np.clip(s2[2] - 1.0, -1.0, 1.0)
        upright_norm = np.clip(s2[390], 0.0, 1.0)
        vel_penalty  = -0.4 * np.tanh(np.linalg.norm(s2[24:27]))
        stand_score  = np.tanh(0.6*height_norm + 0.4*upright_norm + vel_penalty)  # (-1,1)

        # -------- combat score --------
        d_prev     = np.linalg.norm(s1[380:382])
        d_curr     = np.linalg.norm(s2[380:382])
        approach   = np.tanh((d_prev - d_curr) / (arena_r + 1e-6))

        ring_adv   = np.tanh((opp_edge - self_edge) / (arena_r + 1e-6))

        opp_up_p   = s1[390] if s1.size > 390 else 1.0
        opp_up_c   = s_o[390] if s_o.size > 390 else 1.0
        tilt       = np.clip(opp_up_p - opp_up_c, 0.0, 1.0)

        opp_height = s_o[2] if s_o.size > 2 else 2.0
        push_down  = np.clip(1.2 - opp_height, 0.0, 1.0)

        combat_score = np.tanh(0.35*approach + 0.35*ring_adv + 0.15*tilt + 0.15*push_down)  # (-1,1)

        # -------- adaptive dense reward (per‑step) --------
        stand_w  = 0.5 + 0.4 * (1.0 - winning_rate)     # 0.9→0.1
        combat_w = 1.0 - stand_w
        dense_r  = 1.5 * (stand_w * stand_score + combat_w * combat_score)  # ≈ ±1.35

        # --------能量与步长惩罚--------
        energy_pen = -0.1 * float(np.mean(a2 ** 2))     # [-0.1, 0]
        step_pen   = -0.02                              # 常数

        # -------- 稀疏与终局奖励 --------
        sparse_r    = 2.0 * reward_adv - 0.5 * reward_opp
        victory_bonus = {"win": 60.0 * (2.0 - winning_rate), "loss": -60.0}.get(victory_status, 0.0)
        speed_bonus   = 40.0 * time_frac if victory_status == "win" else 0.0

        # -------- total reward --------
        total = dense_r + energy_pen + step_pen + sparse_r + victory_bonus + speed_bonus

        info = {
            "total_reward":   float(total),
            "dense_reward":   float(dense_r),
            "stand_score":    float(stand_score),
            "combat_score":   float(combat_score),
            "stand_weight":   float(stand_w),
            "combat_weight":  float(combat_w),
            "approach":       float(approach),
            "ring_adv":       float(ring_adv),
            "tilt":           float(tilt),
            "push_down":      float(push_down),
            "energy_penalty": float(energy_pen),
            "step_penalty":   float(step_pen),
            "sparse_reward":  float(sparse_r),
            "victory_bonus":  float(victory_bonus),
            "speed_bonus":    float(speed_bonus),
            "winning_rate":   float(winning_rate),
        }
        return float(total), info

    """,
"multicomp/SumoHumans-v0_2": 
"""class RewardCalculator:
    def __init__(self):
        self._stats = {"wins": 0, "games": 0, "steps": 0}

    def compute(self, s1, s2, a2, victory_status, s_o, reward_adv, reward_opp):
        # ---------- persistent counters ----------
        if victory_status in ("win", "loss"):
            self._stats["games"] += 1
            self._stats["wins"] += (victory_status == "win")
        self._stats["steps"] += 1
        progress = min(self._stats["steps"] / 200_000, 1.0)        # 0 → 1

        # ---------- numpy cast ----------
        s1, s2, s_o, a2 = map(lambda x: np.asarray(x, np.float32), (s1, s2, s_o, a2))
        arena_r     = s2[391] if s2.size > 391 else 3.0
        self_edge   = s2[392] if s2.size > 392 else arena_r
        opp_edge    = s2[393] if s2.size > 393 else arena_r
        remain_time = s2[394] if s2.size > 394 else 0.0
        time_frac   = remain_time / 400.0

        # ---------- standing score ----------
        height_n = np.clip(s2[2] - 1.0, -1.0, 1.0)
        up_n     = np.clip(s2[390], 0.0, 1.0)
        vel_pen  = -0.3 * np.tanh(np.linalg.norm(s2[24:27]))
        stand_s  = np.clip(0.6*height_n + 0.4*up_n + vel_pen, -1.0, 1.0)

        # ---------- combat score ----------
        d_prev   = np.linalg.norm(s1[380:382])
        d_curr   = np.linalg.norm(s2[380:382])
        approach = np.tanh((d_prev - d_curr) / (arena_r + 1e-6))

        ring_adv = np.tanh((opp_edge - self_edge) / (arena_r + 1e-6))

        tilt_ang = np.clip((s1[390] if s1.size > 390 else 1.0) -
                           (s_o[390] if s_o.size > 390 else 1.0), 0.0, 1.0)
        tilt_c   = tilt_ang ** 0.5

        push_depth = np.clip(1.0 - (s_o[2] if s_o.size > 2 else 2.0), 0.0, 1.0)

        combat_raw = 0.3*approach + 0.3*ring_adv + 0.2*tilt_c + 0.2*push_depth
        combat_s   = np.clip(combat_raw, -1.0, 1.0)

        # ---------- dense reward ----------
        stand_w  = np.clip(0.6 - 0.4 * progress, 0.15, 0.6)
        combat_w = 1.0 - stand_w
        dense_r  = stand_w * stand_s + combat_w * combat_s            # (-1,1)

        # ---------- penalties ----------
        energy_pen = -0.006 * float(np.sum(a2 ** 2))
        step_pen   = -0.003

        # ---------- sparse / terminal ----------
        sparse_r      = 1.2 * reward_adv - 0.2 * reward_opp
        victory_bonus = {"win": 25.0, "loss": -25.0}.get(victory_status, 0.0)
        speed_bonus   = 10.0 * time_frac if victory_status == "win" else 0.0

        # ---------- total reward ----------
        total = dense_r + energy_pen + step_pen + sparse_r + victory_bonus + speed_bonus

        info = {
            "total_reward":   float(total),
            "dense_reward":   float(dense_r),
            "stand_score":    float(stand_s),
            "combat_score":   float(combat_s),
            "stand_weight":   float(stand_w),
            "combat_weight":  float(combat_w),
            "approach":       float(approach),
            "ring_adv":       float(ring_adv),
            "tilt":           float(tilt_c),
            "push_depth":     float(push_depth),
            "energy_penalty": float(energy_pen),
            "step_penalty":   float(step_pen),
            "sparse_reward":  float(sparse_r),
            "victory_bonus":  float(victory_bonus),
            "speed_bonus":    float(speed_bonus),
            "steps":          int(self._stats["steps"]),
        }
        return float(total), info


    """,
"multicomp/SumoHumans-v0_3": 
"""class RewardCalculator:
    def __init__(self,
                 stand_weight    = 0.4,
                 combat_weight   = 0.6,
                 bonus_scale     = 5.0,   # 终局 ±5（原来 ±25）
                 reward_scale    = 1.0,   # 最终乘以该系数
                 clip_action_pen = 1.0):  # 动作能耗惩罚系数
        self._stats = {"wins": 0, "games": 0, "steps": 0}

        # 固定权重 & 缩放参数
        self.stand_w   = float(stand_weight)
        self.combat_w  = float(combat_weight)
        self.bonus_scl = float(bonus_scale)
        self.rew_scl   = float(reward_scale)
        self.act_pen   = float(clip_action_pen)

    # ------------------------------------------------------------------
    # 计算一次奖励
    # ------------------------------------------------------------------
    def compute(self, s1, s2, a2, victory_status,
                s_oppo,           # 对手上一状态
                reward_adv,       # 原环境给的优势奖励
                reward_opp):      # 对手获得的奖励
        self._update_stats(victory_status)

        # ---------- numpy cast ----------
        s1, s2, s_oppo, a2 = map(lambda x: np.asarray(x, np.float32),
                                 (s1, s2, s_oppo, a2))

        # --------- dense: 站稳（姿态+速度） ----------
        height_n = np.clip(s2[2] - 1.0, -1.0, 1.0)      # 身高归一
        up_n     = np.clip(s2[390], 0.0, 1.0)           # 躯干竖直余弦
        vel_pen  = -0.3 * np.tanh(np.linalg.norm(s2[24:27])/2.0)  # /2 防饱和
        stand_s  = np.clip(0.6*height_n + 0.4*up_n + vel_pen,
                           -1.0, 1.0)                   # (-1,1)

        # --------- dense: 近战（距离+环位+推倒） ----------
        arena_r  = s2[391] if s2.size > 391 else 3.0
        d_prev   = np.linalg.norm(s1[380:382])
        d_curr   = np.linalg.norm(s2[380:382])
        approach = np.tanh((d_prev - d_curr) / (arena_r*0.5 + 1e-6))

        self_edge = s2[392] if s2.size > 392 else arena_r
        opp_edge  = s2[393] if s2.size > 393 else arena_r
        ring_adv  = np.tanh((opp_edge - self_edge) / (arena_r*0.5 + 1e-6))

        tilt_c = np.clip((s1[390] if s1.size > 390 else 1.0) -
                         (s_oppo[390] if s_oppo.size > 390 else 1.0),
                         0.0, 1.0)

        push_depth = np.clip(1.0 - (s_oppo[2] if s_oppo.size > 2 else 2.0),
                             0.0, 1.0)

        combat_raw = 0.3*approach + 0.3*ring_adv + 0.2*tilt_c + 0.2*push_depth
        combat_s   = np.clip(combat_raw, -1.0, 1.0)

        # --------- dense reward ----------
        dense_r = self.stand_w * stand_s + self.combat_w * combat_s   # (-1,1)

        # --------- 小惩罚 ----------
        energy_pen = -self.act_pen * 0.006 * float(np.sum(a2 ** 2))
        step_pen   = -0.002                                           # 常数惩罚

        # --------- sparse / bonus ----------
        sparse_r      = 1.2 * reward_adv - 0.2 * reward_opp
        victory_bonus = {"win": +1.0, "loss": -1.0}.get(victory_status, 0.0)
        speed_bonus   = (0.4 if victory_status == "win" else 0.0)  # 归一化到 0~0.4

        # 统一缩放
        victory_bonus *= self.bonus_scl
        speed_bonus   *= self.bonus_scl

        # --------- total reward ----------
        total = (dense_r + energy_pen + step_pen +
                 sparse_r + victory_bonus + speed_bonus)

        total *= self.rew_scl

        # --------- info dict ----------
        info = dict(
            total_reward   = float(total),
            dense_reward   = float(dense_r),
            stand_score    = float(stand_s),
            combat_score   = float(combat_s),
            stand_weight   = self.stand_w,
            combat_weight  = self.combat_w,
            approach       = float(approach),
            ring_adv       = float(ring_adv),
            tilt           = float(tilt_c),
            push_depth     = float(push_depth),
            energy_penalty = float(energy_pen),
            step_penalty   = float(step_pen),
            sparse_reward  = float(sparse_r),
            victory_bonus  = float(victory_bonus),
            speed_bonus    = float(speed_bonus),
            steps          = int(self._stats["steps"]),
            games          = int(self._stats["games"]),
            win_rate       = float(self._stats["wins"] /
                                   max(self._stats["games"], 1)),
        )
        return float(total), info

    # --------------------------------------------------------------
    def _update_stats(self, victory_status: str):
        if victory_status in ("win", "loss"):
            self._stats["games"] += 1
            self._stats["wins"] += (victory_status == "win")
        self._stats["steps"] += 1
    """,
    "multicomp/SumoHumans-v0": 
"""class RewardCalculator:
    def __init__(self,
                 gamma=0.99,
                 win_bonus=8.0,
                 loss_penalty=-4.0,
                 energy_coef=0.005,
                 step_penalty=-0.001):
        self.gamma        = float(gamma)
        self.win_bonus    = float(win_bonus)
        self.loss_penalty = float(loss_penalty)
        self.energy_coef  = float(energy_coef)
        self.step_pen     = float(step_penalty)
        self._stats       = {"wins": 0, "games": 0, "steps": 0}
        self._ep_steps    = 0

    # --------------------------------------------------------------
    def compute(self, s1, s2, a2, victory_status,
                s_oppo, reward_adv, reward_opp):
        import numpy as np

        # -------- stats / episode step --------
        self._update_stats(victory_status)
        self._ep_steps += 1

        s1, s2, s_oppo, a2 = map(lambda x: np.asarray(x, np.float32),
                                 (s1, s2, s_oppo, a2))

        # -------- adaptive weights (shift from站立→对抗) --------
        win_rate   = self._stats["wins"] / max(self._stats["games"], 1)
        combat_w   = 0.3 + 0.7 * win_rate
        stand_w    = 1.0 - combat_w

        # -------- potential helpers --------
        def stand_phi(s):
            height = np.clip(s[2] - 1.0, -1.0, 1.0)
            upright = np.clip(s[390], 0.0, 1.0)
            vel_pen = -0.5 * np.tanh(np.linalg.norm(s[24:27]) / 2.0)
            return 0.6 * height + 0.4 * upright + vel_pen        # (-1,1)

        def combat_phi(s_self, s_enemy):
            arena_r   = s_self[391] if s_self.size > 391 else 3.0
            dist      = np.linalg.norm(s_self[380:382]) / max(arena_r, 1e-6)
            ring_adv  = np.tanh((s_enemy[393] - s_self[392]) /
                                (arena_r * 0.5 + 1e-6))
            tilt      = np.clip(s_enemy[390] - s_self[390], 0.0, 1.0)
            push      = np.clip(1.0 - s_enemy[2], 0.0, 1.0)
            return -0.6 * dist + 0.2 * ring_adv + 0.1 * tilt + 0.1 * push

        phi1 = stand_w * stand_phi(s1) + combat_w * combat_phi(s1, s_oppo)
        phi2 = stand_w * stand_phi(s2) + combat_w * combat_phi(s2, s_oppo)
        dense_r = self.gamma * phi2 - phi1                                # (-2,2) per step

        # -------- sparse & event rewards --------
        sparse_r = 0.5 * reward_adv - 0.1 * reward_opp
        terminal_bonus = (self.win_bonus if victory_status == "win"
                          else self.loss_penalty if victory_status == "loss"
                          else 0.0)

        # -------- penalties --------
        energy_pen = -self.energy_coef * float(np.sum(a2 ** 2))
        step_pen   = self.step_pen

        # -------- total --------
        total = dense_r + sparse_r + terminal_bonus + energy_pen + step_pen

        info = dict(
            total_reward   = float(total),
            dense_reward   = float(dense_r),
            stand_weight   = float(stand_w),
            combat_weight  = float(combat_w),
            sparse_reward  = float(sparse_r),
            terminal_bonus = float(terminal_bonus),
            energy_penalty = float(energy_pen),
            step_penalty   = float(step_pen),
            win_rate       = float(win_rate),
            steps_global   = int(self._stats["steps"]),
            steps_episode  = int(self._ep_steps),
        )
        return float(total), info

    # --------------------------------------------------------------
    def _update_stats(self, victory_status: str):
        if victory_status in ("win", "loss"):
            self._stats["games"] += 1
            self._stats["wins"] += (victory_status == "win")
            self._ep_steps = 0            # reset per episode
        self._stats["steps"] += 1

    """,
    "multicomp/YouShallNotPassHumans-v0_0":
"""class RewardCalculator:
    def __init__(self,
                 gamma=0.99,
                 win_bonus=8.0,
                 loss_penalty=-4.0,
                 energy_coef=0.005,
                 step_penalty=-0.001):
        self.gamma        = float(gamma)
        self.win_bonus    = float(win_bonus)
        self.loss_penalty = float(loss_penalty)
        self.energy_coef  = float(energy_coef)
        self.step_pen     = float(step_penalty)
        self._stats       = {"wins": 0, "games": 0, "steps": 0}
        self._ep_steps    = 0

    # --------------------------------------------------------------
    def compute(self, s1, s2, a2, victory_status,
                s_oppo, reward_adv, reward_opp):
        import numpy as np

        # -------- stats / episode step --------
        self._update_stats(victory_status)
        self._ep_steps += 1

        s1, s2, s_oppo, a2 = map(lambda x: np.asarray(x, np.float32),
                                 (s1, s2, s_oppo, a2))

        # -------- adaptive weights (shift from站立→阻拦) --------
        win_rate  = self._stats["wins"] / max(self._stats["games"], 1)
        block_w   = 0.8 + 0.2 * win_rate
        stand_w   = 1.0 - block_w

        # -------- potential helpers --------
        def stand_phi(s):
            height  = np.clip((s[2] - 0.3) / 0.7, -1.0, 1.0)   # 0.3~1.0 → (-1,1)
            upright = np.clip(s[23] if s.size > 23 else 1.0, 0.0, 1.0)
            vel_pen = -0.5 * np.tanh(np.linalg.norm(s[24:27]) / 2.0)
            return 0.6 * height + 0.4 * upright + vel_pen       # (-1,1)

        def block_phi(b, w):
            ahead      = np.tanh((b[0] - w[0]) / 1.0)           # blocker ahead on x
            lat_dist   = np.abs(b[1] - w[1])
            lateral    = -np.tanh(lat_dist / 1.0)               # keep close laterally
            walker_vx  = w[24] if w.size > 24 else 0.0
            slow_walk  = -0.5 * np.tanh(max(walker_vx, 0.0))    # reduce fwd speed
            walker_low = np.clip(0.8 - w[2], 0.0, 1.0)          # walker height drop
            return 0.4 * ahead + 0.3 * lateral + 0.2 * slow_walk + 0.1 * walker_low

        phi1 = stand_w * stand_phi(s1) + block_w * block_phi(s1, s_oppo)
        phi2 = stand_w * stand_phi(s2) + block_w * block_phi(s2, s_oppo)
        dense_r = self.gamma * phi2 - phi1                      # (-2,2) per step

        # -------- sparse & terminal rewards --------
        sparse_r = 0.5 * reward_adv - 0.1 * reward_opp
        terminal_bonus = (self.win_bonus if victory_status == "win"
                          else self.loss_penalty if victory_status == "loss"
                          else 0.0)

        # -------- penalties --------
        energy_pen = -self.energy_coef * float(np.sum(a2 ** 2))
        step_pen   = self.step_pen

        # -------- total --------
        total = dense_r + sparse_r + terminal_bonus + energy_pen + step_pen

        info = dict(
            total_reward   = float(total),
            dense_reward   = float(dense_r),
            stand_weight   = float(stand_w),
            block_weight   = float(block_w),
            sparse_reward  = float(sparse_r),
            terminal_bonus = float(terminal_bonus),
            energy_penalty = float(energy_pen),
            step_penalty   = float(step_pen),
            win_rate       = float(win_rate),
            steps_global   = int(self._stats["steps"]),
            steps_episode  = int(self._ep_steps),
        )
        return float(total), info

    # --------------------------------------------------------------
    def _update_stats(self, victory_status: str):
        if victory_status in ("win", "loss"):
            self._stats["games"] += 1
            self._stats["wins"] += (victory_status == "win")
            self._ep_steps = 0            # reset per episode
        self._stats["steps"] += 1
""",
"multicomp/KickAndDefend-v0_0":
"""class RewardCalculator:
    def __init__(
        self,
        gamma: float = 0.98,
        win_bonus: float = 6.0,
        loss_penalty: float = -6.0,
        step_penalty: float = -0.004,
        energy_coef: float = 0.004,
        shield_scale: float = 3.0,
        contact_scale: float = 2.0,
        height_target: float = 1.0,
        ema_alpha: float = 0.03,
    ):
        self.g  = float(gamma)
        self.W  = float(win_bonus)
        self.L  = float(loss_penalty)
        self.Ct = float(step_penalty)
        self.Ce = float(energy_coef)
        self.Ss = float(shield_scale)
        self.Cs = float(contact_scale)
        self.h0 = float(height_target)
        self.a  = float(ema_alpha)

        self.win_rate_ema = 0.0
        self.games        = 0
        self.steps_global = 0
        self.steps_ep     = 0

    # ------------------------------------------------------------
    def compute(self, s1, s2, a2, victory_status,
                s_o, reward_adv, reward_opp):
        import numpy as np

        s1 = np.asarray(s1, np.float32)
        s2 = np.asarray(s2, np.float32)
        a2 = np.asarray(a2, np.float32)
        s_o = np.asarray(s_o, np.float32)

        # ---------- episode / win statistics ----------
        if victory_status in ("win", "loss"):
            self.games += 1
            self.win_rate_ema = (1 - self.a) * self.win_rate_ema + \
                                self.a * float(victory_status == "win")
            self.steps_ep = 0
        self.steps_global += 1
        self.steps_ep     += 1

        # ---------- posture: stand stability ----------
        z = s2[0]
        height_r = np.tanh((z - 0.3) / (self.h0 - 0.3))
        vel_pen  = -np.tanh(np.linalg.norm(s2[22:25]) / 3.0)
        stand_r  = 0.5 * (0.7 * height_r + 0.3 * vel_pen)               # (-0.5,0.5)

        # ---------- shield: ball–goal distance change ----------
        goal_rel1 = s1[381:384]                                          # (dx,dy_l,dy_r)
        goal_rel2 = s2[381:384]
        dist0 = np.linalg.norm(goal_rel1[:2])
        dist1 = np.linalg.norm(goal_rel2[:2])
        shield_r = self.Ss * (np.exp(-dist1) - np.exp(-dist0))           # (-Ss,Ss)

        # ---------- contact: ball moving toward keeper ----------
        ball_rel1 = s1[378:381]
        ball_rel2 = s2[378:381]
        rel_vel   = ball_rel2[:2] - ball_rel1[:2]
        toward_k  = -np.dot(rel_vel, ball_rel2[:2])                      # >0 toward
        contact_r = self.Cs * np.tanh(toward_k)                          # (-Cs,Cs)

        # ---------- dense reward ----------
        dense_r = self.g * (stand_r + shield_r + contact_r)

        # ---------- sparse reward ----------
        sparse_r = 0.8 * (np.tanh(0.2 * reward_adv)
                          - 0.2 * np.tanh(0.2 * reward_opp))

        # ---------- terminal bonus ----------
        term_r = self.W if victory_status == "win" else \
                 self.L if victory_status == "loss" else 0.0

        # ---------- penalties ----------
        energy_pen = -self.Ce * float(np.sum(a2 ** 2))
        step_pen   = self.Ct

        # ---------- total ----------
        raw_total = dense_r + sparse_r + term_r + energy_pen + step_pen
        total     = np.clip(raw_total, -10.0, 10.0)

        info = dict(
            total_reward   = float(total),
            dense_reward   = float(dense_r),
            stand_reward   = float(stand_r),
            shield_reward  = float(shield_r),
            contact_reward = float(contact_r),
            sparse_reward  = float(sparse_r),
            terminal_bonus = float(term_r),
            energy_penalty = float(energy_pen),
            step_penalty   = float(step_pen),
            win_rate_ema   = float(self.win_rate_ema),
            steps_global   = int(self.steps_global),
            steps_episode  = int(self.steps_ep),
        )
        return float(total), info

""",
"multicomp/KickAndDefend-v0_2":
"""class RewardCalculator:
    def __init__(
        self,
        win_bonus: float = 20.0,
        loss_penalty: float = -20.0,
        dx_weight: float = 2.0,
        dx_delta_weight: float = 4.0,
        lateral_weight: float = 1.5,
        energy_weight: float = -0.1,
        opp_speed_weight: float = 0.5,
        step_penalty: float = -0.005,
        adv_diff_weight: float = 0.2,
        clip_limit: float = 25.0,
    ):
        self.W   = win_bonus
        self.L   = loss_penalty
        self.wx  = dx_weight
        self.wdx = dx_delta_weight
        self.wy  = lateral_weight
        self.we  = energy_weight
        self.wo  = opp_speed_weight
        self.ws  = step_penalty
        self.wa  = adv_diff_weight
        self.M   = clip_limit

    # ------------------------------------------------------------
    def compute(self, s1, s2, a2, victory_status,
                s_o, reward_adv, reward_opp):
        import numpy as np

        s1 = np.asarray(s1, dtype=np.float32)
        s2 = np.asarray(s2, dtype=np.float32)
        s_o = np.asarray(s_o, dtype=np.float32)
        a2 = np.asarray(a2, dtype=np.float32)

        # 1. 终局奖励
        terminal = 0.0
        if victory_status == "win":
            terminal = self.W
        elif victory_status == "loss":
            terminal = self.L

        # 2. 球到球门距离 (越远越好)
        dx1, dx2 = s1[381], s2[381]
        r_dx      = self.wx  * np.tanh(dx2 / 5.0)
        r_dx_inc  = self.wdx * (np.tanh(dx2) - np.tanh(dx1))

        # 3. 横向对齐 (越贴近球 y 越好)
        dy_align  = s2[379]
        r_lat     = self.wy * (1.0 - np.tanh(abs(dy_align)))

        # 4. 能量惩罚 (动作扭矩平方和)
        energy    = np.sum(a2 ** 2)
        r_energy  = self.we * np.tanh(energy)

        # 5. 对手前进速度惩罚 (抑制 k icker 向前冲刺)
        opp_speed = s_o[22] if s_o.size > 22 else 0.0
        r_opp     = -self.wo * np.tanh(abs(opp_speed))

        # 6. 基本奖励差异 (利用环境自身奖励信号)
        r_advdiff = self.wa * (reward_adv - reward_opp)

        # 7. 步长惩罚 (鼓励尽快终局)
        r_step    = self.ws

        # 8. 总奖励
        total = (
            terminal +
            r_dx + r_dx_inc +
            r_lat +
            r_energy +
            r_opp +
            r_advdiff
        )
        total = float(np.clip(total, -self.M, self.M))

        info = {
            "total_reward": total,
            "terminal": float(terminal),
            "r_dx": float(r_dx),
            "r_dx_inc": float(r_dx_inc),
            "r_lat": float(r_lat),
            "r_energy": float(r_energy),
            "r_opp": float(r_opp),
            "r_advdiff": float(r_advdiff),
            "r_step": float(r_step),
        }
        return total, info

""",
"multicomp/KickAndDefend-v0":
"""import numpy as np

class RewardCalculator:
    def __init__(
        self,
        win_bonus: float = 2.0,
        loss_penalty: float = -2.0,
        dist_weight: float = 0.02,
        delta_weight: float = 0.03,
        align_weight: float = 0.015,
        threat_weight: float = 0.1,
        threat_radius: float = 2.0,
        gate_penalty_once: float = -0.1,
        stance_penalty_once: float = -0.5,
        energy_weight: float = -0.001,
        adv_diff_weight: float = 0.4,
        time_penalty: float = -0.001,
        clip_limit: float = 115.0,
        align_sigma: float = 0.5,
    ):
        self.W   = win_bonus
        self.L   = loss_penalty
        self.wd  = dist_weight
        self.wΔ  = delta_weight
        self.wa  = align_weight
        self.wt  = threat_weight
        self.rt  = threat_radius
        self.wg  = gate_penalty_once
        self.ws  = stance_penalty_once
        self.we  = energy_weight
        self.wr  = adv_diff_weight
        self.wp  = time_penalty
        self.M   = clip_limit
        self.sig = align_sigma

    def compute(self, s1, s2, a2, victory_status, s_o, reward_adv, reward_opp):
        s1 = np.asarray(s1, dtype=np.float32)
        s2 = np.asarray(s2, dtype=np.float32)
        a2 = np.asarray(a2, dtype=np.float32)

        terminal = self.W if victory_status == "win" else (self.L if victory_status == "loss" else 0.0)

        dx1 = max(0.0, float(s1[381]))
        dx2 = max(0.0, float(s2[381]))
        r_dist  = self.wd * np.tanh(dx2 / 20.0)
        r_delta = self.wΔ * np.tanh((dx2 - dx1) / 4.0)

        dy = float(s2[379])
        r_align = self.wa * np.exp(-dy * dy / (2 * self.sig * self.sig))

        left_gap = float(s2[382])
        right_gap = float(s2[383])
        between = (left_gap * right_gap) < 0.0
        if between and dx2 <= self.rt:
            r_threat = self.wt * (self.rt - dx2) / self.rt
            r_gate   = self.wg
        else:
            r_threat = 0.0
            r_gate   = 0.0

        z1 = float(s1[0])
        z2 = float(s2[0])
        r_stance = self.ws if (z1 >= 0.75 and z2 < 0.75) else 0.0

        r_energy  = self.we * np.sum(a2 * a2)
        r_advdiff = self.wr * (reward_adv - reward_opp)
        r_time    = self.wp

        total = (
            terminal + r_dist + r_delta + r_align +
            r_threat + r_gate + r_stance +
            r_energy + r_advdiff
        )
        total = float(np.clip(total, -self.M, self.M))

        info = {
            "total_reward": total,
            "terminal": terminal,
            "r_dist": r_dist,
            "r_delta": r_delta,
            "r_align": r_align,
            "r_threat": r_threat,
            "r_gate": r_gate,
            "r_stance": r_stance,
            "r_energy": r_energy,
            "r_advdiff": r_advdiff,
            "r_time": r_time,
        }
        return total, info


""",
"multicomp/KickAndDefend-v0_1":
"""class RewardCalculator:
    def __init__(
        self,
        win_bonus: float = 10.0,
        loss_penalty: float = -10.0,
        dx_scale: float = 0.5,
        dy_scale: float = 0.03,
        step_penalty: float = -0.0005,
        clip_limit: float = 15.0,
    ):
        self.W  = win_bonus
        self.L  = loss_penalty
        self.Sx = dx_scale
        self.Sy = dy_scale
        self.Ct = step_penalty
        self.M  = clip_limit

    # ------------------------------------------------------------
    def compute(self, s1, s2, a2, victory_status,
                s_o, reward_adv, reward_opp):
        import numpy as np
        s2 = np.asarray(s2, np.float32)

        # 1. 终局奖励
        term_r = reward_adv - reward_opp

        # 2. 前后距离惩罚 (球离球门线)
        dx_goal = s2[381]                       # goal_x - ball_x
        guard_x = -self.Sx * np.tanh(max(0.0, 1.0 - dx_goal))

        # 3. 横向对齐惩罚 (keeper 与球 y 误差)
        dy_align = s2[379]                      # ball_y - keeper_y
        guard_y  = -self.Sy * np.tanh(abs(dy_align) / 0.5)

        # 4. 总奖励
        total = term_r + guard_x + guard_y
        total = float(np.clip(total, -self.M, self.M))

        info = dict(
            total_reward = total,
            term_r       = float(term_r),
            guard_x      = float(guard_x),
            guard_y      = float(guard_y),
        )
        return total, info


""",
"multicomp/YouShallNotPassHumans-v0_1":
"""class RewardCalculator:
    def __init__(self,
                 gamma=0.995,           # discount for shaping
                 term_scale=25.0,       # amplify ±10 → ±250
                 shaping_scale=6.0,     # per-step dense range ≈ ±6
                 energy_coef=0.002,
                 step_penalty=-0.15/300,  # ≈ -0.15 over 300 steps
                 fall_penalty=-5.0):
        self.g   = gamma
        self.ts  = term_scale
        self.ss  = shaping_scale
        self.ec  = energy_coef
        self.sp  = step_penalty
        self.fp  = fall_penalty
        self.stat = {"wins": 0, "games": 0}

    # ------------------------------------------------------------
    def compute(self, s1, s2, a2, victory_status,
                s_o, reward_adv, reward_opp):
        import numpy as np

        b1, b2, w2, a2 = map(lambda x: np.asarray(x, np.float32),
                             (s1, s2, s_o, a2))

        # ---------- dense shaping (φ difference) ----------
        def phi(b, w):
            x_gap  = np.tanh(b[0] - w[0])                 # block ahead
            lat    = -np.tanh(np.abs(b[1] - w[1]))        # align laterally
            vwx    = w[24] if w.size > 24 else 0.0
            slow   = -np.tanh(-vwx)                       # slow walker
            trip   = np.clip(0.8 - w[2], 0.0, 1.0)        # make walker low
            return 0.45*x_gap + 0.25*lat + 0.2*slow + 0.1*trip

        dense_r = self.ss * (self.g*phi(b2, w2) - phi(b1, w2))

        # ---------- penalties ----------
        energy_pen = -self.ec * float(np.sum(a2**2))
        step_pen   = self.sp
        fall_pen   = self.fp if b2[2] < 0.3 else 0.0

        # ---------- terminal / sparse ----------
        term_r   = self.ts * reward_adv
        sparse_r = 0.5*reward_adv - 0.05*reward_opp        # small mid-game aid

        # ---------- stats ----------
        if victory_status in ("win", "loss"):
            self.stat["games"] += 1
            self.stat["wins"]  += (victory_status == "win")

        total = dense_r + energy_pen + step_pen + fall_pen + term_r + sparse_r

        return float(total), {
            "total_reward": float(total),
            "dense_reward": float(dense_r),
            "terminal":     float(term_r),
            "sparse":       float(sparse_r),
            "energy_pen":   float(energy_pen),
            "step_pen":     float(step_pen),
            "fall_pen":     float(fall_pen),
            "win_rate":     self.stat["wins"] / max(self.stat["games"], 1)
        }

""",
"multicomp/YouShallNotPassHumans-v0_2":
"""class RewardCalculator:
    def __init__(self,
                 gamma=0.99,
                 win_bonus=8.0,
                 loss_penalty=-4.0,
                 energy_coef=0.005,
                 step_penalty=-0.001):
        self.gamma        = float(gamma)
        self.win_bonus    = float(win_bonus)
        self.loss_penalty = float(loss_penalty)
        self.energy_coef  = float(energy_coef)
        self.step_pen     = float(step_penalty)
        self._stats       = {"wins": 0, "games": 0, "steps": 0}
        self._ep_steps    = 0

    # --------------------------------------------------------------
    def compute(self, s1, s2, a2, victory_status,
                s_oppo, reward_adv, reward_opp):
        import numpy as np

        # -------- stats / episode step --------
        self._update_stats(victory_status)
        self._ep_steps += 1

        s1, s2, s_oppo, a2 = map(lambda x: np.asarray(x, np.float32),
                                 (s1, s2, s_oppo, a2))

        # -------- adaptive weights (站立 ↔ 阻拦) --------
        win_rate = self._stats["wins"] / max(self._stats["games"], 1)
        win_rate = 0.7
        block_w  = 0.3 + 0.7 * win_rate
        stand_w  = 1.0 - block_w

        # -------- potential helpers --------
        def stand_phi(s):
            height  = np.clip((s[2] - 0.3) / 0.7, -1.0, 1.0)
            upright = np.clip(s[23] if s.size > 23 else 1.0, 0.0, 1.0)
            vel_pen = -0.5 * np.tanh(np.linalg.norm(s[24:27]) / 2.0)
            return 0.6 * height + 0.4 * upright + vel_pen

        def block_phi(b, w):
            ahead      = np.tanh((w[0] - b[0]) / 1.0)           # 目标 x 最小方向
            lat_dist   = np.abs(b[1] - w[1])
            lateral    = -np.tanh(lat_dist / 1.0)
            walker_vx  = w[24] if w.size > 24 else 0.0
            slow_w     = -0.5 * np.tanh(max(walker_vx, 0.0))
            walker_low = np.clip(0.8 - w[2], 0.0, 1.0)
            return 0.4 * ahead + 0.3 * lateral + 0.2 * slow_w + 0.1 * walker_low

        phi1 = stand_w * stand_phi(s1) + block_w * block_phi(s1, s_oppo)
        phi2 = stand_w * stand_phi(s2) + block_w * block_phi(s2, s_oppo)
        dense_r = self.gamma * phi2 - phi1

        # -------- sparse & terminal rewards --------
        sparse_r = 0.5 * reward_adv - 0.1 * reward_opp
        terminal_bonus = (self.win_bonus if victory_status == "win"
                          else self.loss_penalty if victory_status == "loss"
                          else 0.0)

        # -------- penalties --------
        energy_pen = -self.energy_coef * float(np.sum(a2 ** 2))
        step_pen   = self.step_pen

        # -------- total --------
        total = dense_r + sparse_r + terminal_bonus + energy_pen

        info = dict(
            total_reward   = float(total),
            dense_reward   = float(dense_r),
            stand_weight   = float(stand_w),
            block_weight   = float(block_w),
            sparse_reward  = float(sparse_r),
            terminal_bonus = float(terminal_bonus),
            energy_penalty = float(energy_pen),
            step_penalty   = float(step_pen),
            win_rate       = float(win_rate),
            steps_global   = int(self._stats["steps"]),
            steps_episode  = int(self._ep_steps),
        )
        return float(total), info

    # --------------------------------------------------------------
    def _update_stats(self, victory_status: str):
        if victory_status in ("win", "loss"):
            self._stats["games"] += 1
            self._stats["wins"]  += (victory_status == "win")
            self._ep_steps = 0
        self._stats["steps"] += 1

""",
"multicomp/YouShallNotPassHumans-v0":
"""class RewardCalculator:
    def __init__(
        self,
        gamma: float = 0.995,
        dense_scale: float = 12.0,
        win_bonus: float = 15.0,
        loss_penalty: float = -6.0,
        energy_coef: float = 0.0008,
        step_penalty: float = -0.0005,
        ema_alpha: float = 0.1,
    ):
        self.gamma = float(gamma)
        self.dense_scale = float(dense_scale)
        self.win_bonus = float(win_bonus)
        self.loss_penalty = float(loss_penalty)
        self.energy_coef = float(energy_coef)
        self.step_penalty = float(step_penalty)
        self.ema_alpha = float(ema_alpha)

        self._stats = {"wins": 0, "games": 0, "steps": 0}
        self._ep_steps = 0
        self._ema_win_rate = 0.0

    # --------------------------------------------------------------
    def compute(
        self,
        s1,
        s2,
        a2,
        victory_status,
        s_o,
        reward_adv,
        reward_opp,
    ):
        s1, s2, s_o, a2 = map(lambda x: np.asarray(x, np.float32), (s1, s2, s_o, a2))
        self._update_stats(victory_status)
        self._ep_steps += 1

        # ----------- adaptive weights ----------------
        stand_w = np.clip(0.7 * (1.0 - self._ema_win_rate), 0.3, 0.8)
        block_w = 1.0 - stand_w

        # ----------- potentials ----------------------
        def _stand_phi(s):
            height = np.clip((s[2] - 0.3) / 0.7, 0.0, 1.0)
            vel_pen = -0.3 * np.tanh(np.linalg.norm(s[24:27]))
            return 0.8 * height + vel_pen

        def _block_phi(b, w):
            ahead = np.tanh((w[0] - b[0]) * 2.0)         # blocker in front → +
            lateral = -np.tanh(np.abs(b[1] - w[1]) * 1.5)
            prog_pen = -np.tanh(-w[0])                   # walker接近终点 → -
            return 0.5 * ahead + 0.3 * lateral + 0.2 * prog_pen

        phi1 = stand_w * _stand_phi(s1) + block_w * _block_phi(s1, s_o)
        phi2 = stand_w * _stand_phi(s2) + block_w * _block_phi(s2, s_o)
        dense_r = self.dense_scale * (self.gamma * phi2 - phi1)

        # ----------- sparse & terminal ----------------
        sparse_r = 0.3 * reward_adv - 0.1 * reward_opp
        terminal_r = (
            self.win_bonus if victory_status == "win"
            else self.loss_penalty if victory_status == "loss"
            else 0.0
        )

        # ----------- penalties -----------------------
        energy_pen = -self.energy_coef * float(np.sum(a2 ** 2))
        step_pen = self.step_penalty

        # ----------- total reward --------------------
        total = dense_r + sparse_r + terminal_r + energy_pen

        info = dict(
            total_reward=float(total),
            dense_reward=float(dense_r),
            sparse_reward=float(sparse_r),
            terminal_reward=float(terminal_r),
            energy_penalty=float(energy_pen),
            step_penalty=float(step_pen),
            stand_weight=float(stand_w),
            block_weight=float(block_w),
            ema_win_rate=float(self._ema_win_rate),
            steps_global=int(self._stats["steps"]),
            steps_episode=int(self._ep_steps),
        )
        return float(total), info

    # --------------------------------------------------------------
    def _update_stats(self, victory_status: str):
        if victory_status in ("win", "loss"):
            self._stats["games"] += 1
            self._stats["wins"] += (victory_status == "win")
            # EMA 更新
            current_win_rate = self._stats["wins"] / self._stats["games"]
            self._ema_win_rate = (
                self.ema_alpha * current_win_rate
                + (1.0 - self.ema_alpha) * self._ema_win_rate
            )
            self._ep_steps = 0
        self._stats["steps"] += 1

""",
    }



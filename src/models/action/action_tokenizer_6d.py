"""
action_tokenizer_6d.py

适配6维动作空间的Action Tokenizer (xyz + euler angles, 无gripper)
基于SpatialVLA的action_tokenizer.py修改
"""
from typing import List, Union, Dict, Optional
import numpy as np
from transformers import PreTrainedTokenizerBase
from scipy.stats import norm
import torch

ACTION_TOKEN = '<ACTION{:05d}>'

class TranslationTokenizer:
    """3D平移动作的令牌化器"""
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_bins: Dict,
        bin_policy: Optional[Dict] = None,
        use_spherical: bool = True,
    ):
        self.tokenizer = tokenizer
        self.num_theta_bins = num_bins["theta_bins"]
        self.num_phi_bins = num_bins["phi_bins"]
        self.num_r_bins = num_bins["r_bins"]
        self.use_spherical = use_spherical

        # for indexing
        self.NP = self.num_phi_bins * self.num_r_bins

        # add special action tokens to language tokenizer
        self._vocab_size = self.num_theta_bins * self.num_phi_bins * self.num_r_bins
        token_list = [ACTION_TOKEN.format(i) for i in range(self._vocab_size)]
        self.token_array = np.array(token_list)

        num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)
        print(f"[6D] Add {num_new_tokens} TRANSLATION TOKENS, tokenizer vocab size {self.tokenizer.vocab_size}")

        self.token_start_idx = self.tokenizer.convert_tokens_to_ids(self.token_array[0])
        self.token_end_idx = self.tokenizer.convert_tokens_to_ids(self.token_array[-1])
        self.set_bins(bin_policy)

    def set_bins(self, bin_policy):
        self.theta_bins = np.array(bin_policy["theta_bins"])
        self.phi_bins = np.array(bin_policy["phi_bins"])
        self.r_bins = np.array(bin_policy["r_bins"])

    def cartesian_to_spherical(self, x, y, z):
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)  # polar angle
        phi = np.arctan2(y, x)  # azimuthal angle
        r = np.sqrt(x**2 + y**2 + z**2)
        return theta, phi, r

    def spherical_to_cartesian(self, theta, phi, r):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def __call__(self, action: np.ndarray) -> List[str]:
        """Discretize continuous actions to tokens.
        action: np.ndarray, (n, 3), continuous actions in Cartesian coordinates.
        return: np.ndarray, (n,), tokens.
        """
        if self.use_spherical:
            theta, phi, r = self.cartesian_to_spherical(action[:, 0], action[:, 1], action[:, 2])
        else:
            theta, phi, r = action[:, 0], action[:, 1], action[:, 2]

        disc_theta = np.digitize(theta, self.theta_bins[1:-1])
        disc_phi = np.digitize(phi, self.phi_bins[1:-1])
        disc_r = np.digitize(r, self.r_bins[1:-1])
        ids = disc_theta * self.NP + disc_phi * self.num_r_bins + disc_r
        return self.token_array[ids]

    def decode_token_ids_to_actions(self, action_token_id: np.ndarray) -> np.ndarray:
        """decode token ids to continuous actions.
        action_token_id: np.ndarray, (n,), token ids.
        return: np.ndarray, (n, 3), continuous actions
        """
        action_token_id = np.clip(action_token_id, self.token_start_idx, self.token_end_idx)
        ids = action_token_id - self.token_start_idx
        disc_theta, disc_phi, disc_r = ids // self.NP, (ids % self.NP) // self.num_r_bins, ids % self.num_r_bins

        theta = 0.5 * (self.theta_bins[disc_theta] + self.theta_bins[disc_theta + 1])
        phi = 0.5 * (self.phi_bins[disc_phi] + self.phi_bins[disc_phi + 1])
        r = 0.5 * (self.r_bins[disc_r] + self.r_bins[disc_r + 1])

        x, y, z = self.spherical_to_cartesian(theta, phi, r) if self.use_spherical else (theta, phi, r)
        x, y, z = np.clip([x, y, z], -1, 1)
        return np.stack((x, y, z), axis=1)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class RotationTokenizer:
    """3D旋转动作的令牌化器 (Euler angles)"""
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_bins: Dict,
        bin_policy: Optional[Dict] = None,
        array_begin_idx=None,
    ):
        self.tokenizer = tokenizer
        self.num_roll_bins = num_bins["roll_bins"]
        self.num_pitch_bins = num_bins["pitch_bins"]
        self.num_yaw_bins = num_bins["yaw_bins"]
        self.array_begin_idx = array_begin_idx

        # for indexing
        self.NP = self.num_pitch_bins * self.num_yaw_bins

        # add special action tokens to language tokenizer
        self._vocab_size = self.num_roll_bins * self.num_pitch_bins * self.num_yaw_bins
        token_list = [ACTION_TOKEN.format(i + self.array_begin_idx) for i in range(self._vocab_size)]
        self.token_array = np.array(token_list)

        num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)
        print(f"[6D] Add {num_new_tokens} ROTATION TOKENS, tokenizer vocab size {self.tokenizer.vocab_size}")

        self.token_start_idx = self.tokenizer.convert_tokens_to_ids(self.token_array[0])
        self.token_end_idx = self.tokenizer.convert_tokens_to_ids(self.token_array[-1])
        self.set_bins(bin_policy)

    def set_bins(self, bin_policy):
        self.roll_bins = np.array(bin_policy["roll_bins"])
        self.pitch_bins = np.array(bin_policy["pitch_bins"])
        self.yaw_bins = np.array(bin_policy["yaw_bins"])

    def __call__(self, action: np.ndarray) -> List[str]:
        """Discretize continuous actions to tokens.
        action: np.ndarray, (n, 3), continuous rotation actions (roll, pitch, yaw).
        return: np.ndarray, (n,), tokens.
        """
        roll, pitch, yaw = action[:, 0], action[:, 1], action[:, 2]
        disc_roll = np.clip(np.digitize(roll, self.roll_bins) - 1, 0, self.num_roll_bins - 1)
        disc_pitch = np.clip(np.digitize(pitch, self.pitch_bins) - 1, 0, self.num_pitch_bins - 1)
        disc_yaw = np.clip(np.digitize(yaw, self.yaw_bins) - 1, 0, self.num_yaw_bins - 1)

        ids = disc_roll * self.NP + disc_pitch * self.num_yaw_bins + disc_yaw
        return self.token_array[ids]

    def decode_token_ids_to_actions(self, action_token_id: Union[np.int64, np.ndarray]) -> np.ndarray:
        """decode token ids to continuous actions.
        action_token_id: np.ndarray, (n,), token ids.
        return: np.ndarray, (n, 3), continuous actions
        """
        action_token_id = np.clip(action_token_id, a_min=self.token_start_idx, a_max=self.token_end_idx)
        ids = action_token_id - self.token_start_idx
        disc_roll, disc_pitch, disc_yaw = ids // self.NP, (ids % self.NP) // self.num_yaw_bins, ids % self.num_yaw_bins

        roll = 0.5 * (self.roll_bins[disc_roll] + self.roll_bins[disc_roll + 1])
        pitch = 0.5 * (self.pitch_bins[disc_pitch] + self.pitch_bins[disc_pitch + 1])
        yaw = 0.5 * (self.yaw_bins[disc_yaw] + self.yaw_bins[disc_yaw + 1])
        return np.stack((roll, pitch, yaw), axis=1)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class SpatialActionTokenizer6D:
    """
    6维动作令牌化器 (xyz + roll/pitch/yaw)
    输入: (n, 6) 连续动作
    输出: (n, 2) 令牌 [translation_token, rotation_token]
    """
    range_bins = {
        "translation": {
            "theta_bins": (0.0, np.pi),
            "phi_bins": (-np.pi, np.pi),
            "r_bins": (0.0, np.sqrt(3)),
        },
        "rotation": {
            "roll_bins": (-1.0, 1.0),
            "pitch_bins": (-1.0, 1.0),
            "yaw_bins": (-1.0, 1.0),
        },
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_bins: Dict,
        gs_params: Dict = None,
        bin_policy: Dict = None,
        use_spherical: bool = True,
        min_sigma: float = 0.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            num_bins: {"translation": {"theta_bins": 16, "phi_bins": 32, "r_bins": 8},
                      "rotation": {"roll_bins": 16, "pitch_bins": 16, "yaw_bins": 16}}
            gs_params: Gaussian参数，用于自适应bin策略
            bin_policy: 预定义的bin边界
            use_spherical: 是否使用球面坐标
        """
        self.tokenizer = tokenizer
        self.min_action, self.max_action = min_action, max_action
        self.num_bins = num_bins
        self.min_sigma = min_sigma

        # set bin policy
        self.bin_policy = bin_policy if bin_policy else self.get_bin_policy(gs_params, self.min_sigma)

        self.translation_tokenizer = TranslationTokenizer(
            self.tokenizer,
            self.num_bins["translation"],
            self.bin_policy["translation"],
            use_spherical=use_spherical
        )

        self.rotation_tokenizer = RotationTokenizer(
            self.tokenizer,
            self.num_bins["rotation"],
            self.bin_policy["rotation"],
            array_begin_idx=self.translation_tokenizer.vocab_size,
        )

        self._vocab_size = self.translation_tokenizer.vocab_size + self.rotation_tokenizer.vocab_size
        print(f"[6D] Total action vocab size: {self._vocab_size}")

    def __call__(self, action: np.ndarray) -> np.ndarray:
        """Discretize continuous actions to tokens.
        Args:
            action: np.ndarray, (n, 6), continuous actions [x, y, z, roll, pitch, yaw]
        Returns:
            np.ndarray, (n, 2), tokens [translation_token, rotation_token]
        """
        if len(action.shape) == 1:
            assert action.shape[0] == 6, f"action dim mismatch, expected 6, got {action.shape}"
            action = action.reshape(1, 6)
        assert action.shape[1] == 6, f"action dim mismatch, expected (n, 6), got {action.shape}"

        action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
        trans_tokens = self.translation_tokenizer(action[:, :3])  # (n,)
        rot_tokens = self.rotation_tokenizer(action[:, 3:6])      # (n,)
        return np.stack((trans_tokens, rot_tokens), axis=1)       # (n, 2)

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """decode token ids to continuous actions.
        Args:
            action_token_ids: np.ndarray, (n, 2), token ids [trans_id, rot_id]
        Returns:
            np.ndarray, (n, 6), continuous actions [x, y, z, roll, pitch, yaw]
        """
        if len(action_token_ids.shape) == 1:
            assert action_token_ids.shape[0] == 2, f"expected 2 tokens, got {action_token_ids.shape[0]}"
            action_token_ids = action_token_ids.reshape(1, 2)
        assert action_token_ids.shape[1] == 2, f"expected (n, 2), got {action_token_ids.shape}"

        trans_action = self.translation_tokenizer.decode_token_ids_to_actions(action_token_ids[:, 0])  # (n, 3)
        rot_action = self.rotation_tokenizer.decode_token_ids_to_actions(action_token_ids[:, 1])      # (n, 3)
        return np.concatenate((trans_action, rot_action), axis=1)  # (n, 6)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def action_token_begin_idx(self) -> int:
        return self.translation_tokenizer.token_start_idx

    @property
    def action_tokens(self) -> List[str]:
        """返回所有动作令牌字符串"""
        return [ACTION_TOKEN.format(i) for i in range(self._vocab_size)]

    def get_bin_policy(self, gs_params=None, min_sigma=0.0):
        """生成bin策略"""
        bin_policy = {
            "translation": {"theta_bins": None, "phi_bins": None, "r_bins": None},
            "rotation": {"roll_bins": None, "pitch_bins": None, "yaw_bins": None}
        }
        if gs_params is None:
            # 使用均匀分布
            for bin_type in self.range_bins.keys():
                for bin_key in self.range_bins[bin_type].keys():
                    bin_policy[bin_type][bin_key] = np.linspace(
                        *self.range_bins[bin_type][bin_key],
                        self.num_bins[bin_type][bin_key] + 1
                    )
            print(f"[6D] Using uniform bin grids")
        else:
            # 使用高斯分布自适应bin
            for bin_type in self.range_bins.keys():
                for bin_key in self.range_bins[bin_type].keys():
                    mu = gs_params[bin_key.split("_")[0].lower()]["mu"]
                    sigma = max(gs_params[bin_key.split("_")[0].lower()]["sigma"], min_sigma)
                    bin_bound_prob = np.linspace(
                        norm.cdf(self.range_bins[bin_type][bin_key][0], loc=mu, scale=sigma),
                        norm.cdf(self.range_bins[bin_type][bin_key][1], loc=mu, scale=sigma),
                        self.num_bins[bin_type][bin_key] + 1,
                    )
                    bin_boundary = norm.ppf(bin_bound_prob, loc=mu, scale=sigma)
                    bin_policy[bin_type][bin_key] = np.clip(
                        bin_boundary,
                        self.range_bins[bin_type][bin_key][0],
                        self.range_bins[bin_type][bin_key][1],
                    ).tolist()
            print(f"[6D] Using gaussian-adaptive bin grids")
        return bin_policy

    def spatial_embedding_adaption(self, gs_params, embeddings: torch.nn.Embedding, min_sigma=0.0, adpt_feature=False):
        """
        自适应调整spatial embedding以适应新的数据分布
        Args:
            gs_params: 新数据的高斯分布参数
            embeddings: 模型的spatial_embed_tokens
            min_sigma: 最小标准差
            adpt_feature: 是否通过插值调整embedding特征
        """
        from scipy.interpolate import griddata

        # 生成新的bin policy
        new_policy = self.get_bin_policy(gs_params, min_sigma=min_sigma)

        # 获取旧的和新的网格
        trans_grids0, rot_grids0 = self.get_norm_meshgrid(self.bin_policy)
        trans_grids1, rot_grids1 = self.get_norm_meshgrid(new_policy)

        # 更新bin policy和tokenizer
        print("[6D] Updating bin policy...")
        self.bin_policy = new_policy
        self.min_sigma = min_sigma
        self.translation_tokenizer.set_bins(new_policy["translation"])
        self.rotation_tokenizer.set_bins(new_policy["rotation"])

        if adpt_feature:
            print("[6D] Adapting embedding features...")
            emb_data = embeddings.weight.data  # (S, E)
            _, E = emb_data.shape

            # Translation embedding adaptation
            m, n, k = (self.num_bins["translation"][key] for key in ["theta_bins", "phi_bins", "r_bins"])
            N = m * n * k
            trans_emb_data = emb_data[:N].reshape(m, n, k, -1).permute(3, 0, 1, 2)
            pad_emb = torch.nn.functional.pad(trans_emb_data, (1, 1, 1, 1, 1, 1), "replicate").permute(1, 2, 3, 0).reshape(-1, E)
            adpt_trans_emb = griddata(trans_grids0, pad_emb.float().cpu().numpy(), trans_grids1, method='linear')
            adpt_trans_emb = adpt_trans_emb.reshape(m+2, n+2, k+2, E)[1:-1, 1:-1, 1:-1]

            # Rotation embedding adaptation
            m1, n1, k1 = (self.num_bins["rotation"][key] for key in ["roll_bins", "pitch_bins", "yaw_bins"])
            M = m1 * n1 * k1
            rot_emb_data = emb_data[N : N + M].reshape(m1, n1, k1, -1).permute(3, 0, 1, 2)
            pad_emb = torch.nn.functional.pad(rot_emb_data, (1, 1, 1, 1, 1, 1), "replicate").permute(1, 2, 3, 0).reshape(-1, E)
            adpt_rot_emb = griddata(rot_grids0, pad_emb.float().cpu().numpy(), rot_grids1, method='linear')
            adpt_rot_emb = adpt_rot_emb.reshape(m1+2, n1+2, k1+2, E)[1:-1, 1:-1, 1:-1]

            # Update embeddings
            device, dtype = embeddings.weight.data.device, embeddings.weight.data.dtype
            embeddings.weight.data[:N] = torch.tensor(adpt_trans_emb.reshape(-1, E), device=device, dtype=dtype)
            embeddings.weight.data[N:N+M] = torch.tensor(adpt_rot_emb.reshape(-1, E), device=device, dtype=dtype)
            print("[6D] Spatial embedding adaptation completed!")

    def get_norm_meshgrid(self, bin_policy):
        """生成归一化的网格用于插值"""
        grids = []
        policy = {k1: {k2: np.array(v2) for k2, v2 in v1.items()} for k1, v1 in bin_policy.items()}

        for bin_type in self.range_bins.keys():
            bounds = []
            for bin_key in self.range_bins[bin_type].keys():
                minb, maxb = self.range_bins[bin_type][bin_key][0], self.range_bins[bin_type][bin_key][1]
                bin_boundary = policy[bin_type][bin_key]
                bin_center = (bin_boundary[:-1] + bin_boundary[1:]) / 2
                bin_center = np.concatenate([np.array([minb]), bin_center, np.array([maxb])])
                bin_center = (bin_center - minb) / (maxb - minb)
                bounds.append(bin_center)

            grid_x, grid_y, grid_z = np.meshgrid(*bounds)
            grids.append(np.stack([grid_x, grid_y, grid_z], -1).reshape(-1, 3))

        return grids[0], grids[1]  # (N, 3)

import numpy as np

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py


# Simple replay buffer
# initial:
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.next_idx = 0

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.max_size

    def sample(self, batch_size=100, gdroweight=None):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        if gdroweight is not None:
            x, y, u, r, d, ret = [], [], [], [], [], []

            for i in ind:
                X, Y, U, R, D, Ret = self.storage[i]
                x.append(np.array(X, copy=False))
                y.append(np.array(Y, copy=False))
                u.append(np.array(U, copy=False))
                r.append(np.array(R, copy=False))
                d.append(np.array(D, copy=False))
                ret.append(np.array(Ret, copy=False))

            return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), np.array(ret).reshape(-1, 1)
        else:
            x, y, u, r, d = [], [], [], [], []

            for i in ind:
                X, Y, U, R, D = self.storage[i]
                x.append(np.array(X, copy=False))
                y.append(np.array(Y, copy=False))
                u.append(np.array(U, copy=False))
                r.append(np.array(R, copy=False))
                d.append(np.array(D, copy=False))

            return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

# import numpy as np


# class ReplayBuffer(object):
#     """
#     分层 replay buffer:
#     - anchor_storage:保留 warm-up 阶段的均匀采样经验,永不覆盖
#     - sliding_storage:保留最近的 sampler 经验,FIFO 覆盖

#     Sample 时按 anchor_ratio 在 batch 内混合两部分经验,
#     确保 critic 永远见到整个任务空间的经验。
#     """

#     def __init__(self,
#                  anchor_size=200_000,      # 匹配 warm-up 步数
#                  sliding_size=1_000_000,   # 保持和原来一样
#                  anchor_fill_steps=200_000):
#         self.anchor_storage  = []
#         self.sliding_storage = []
#         self.anchor_size       = int(anchor_size)
#         self.sliding_size      = int(sliding_size)
#         self.anchor_fill_steps = int(anchor_fill_steps)

#         self.sliding_next_idx = 0
#         self.total_added      = 0
#         self._anchor_frozen   = False

#     # ── 兼容旧接口 ─────────────────────────────────────────────
#     @property
#     def storage(self):
#         """保持对 len(buffer.storage) 等旧用法的兼容。"""
#         return self.anchor_storage + self.sliding_storage

#     @property
#     def next_idx(self):
#         return self.sliding_next_idx
#     # ─────────────────────────────────────────────────────────

#     def add(self, data):
#         if not self._anchor_frozen:
#             # Warm-up 期:经验全部塞进 anchor
#             self.anchor_storage.append(data)
#             if (len(self.anchor_storage) >= self.anchor_size or
#                 self.total_added        >= self.anchor_fill_steps):
#                 self._anchor_frozen = True
#         else:
#             # Sampler 期:FIFO 覆盖 sliding
#             if self.sliding_next_idx >= len(self.sliding_storage):
#                 self.sliding_storage.append(data)
#             else:
#                 self.sliding_storage[self.sliding_next_idx] = data
#             self.sliding_next_idx = (self.sliding_next_idx + 1) % self.sliding_size

#         self.total_added += 1

#     def sample(self, batch_size=100, gdroweight=None, anchor_ratio=0.3):
#         """
#         anchor_ratio:batch 里有多大比例从 anchor 采样。
#         默认 0.3 即 30% 从 anchor,70% 从 sliding。
#         """
#         # anchor 为空时(训练刚开始),全部从 sliding 采
#         if len(self.anchor_storage) == 0:
#             n_anchor  = 0
#             n_sliding = batch_size
#         else:
#             n_anchor  = int(batch_size * anchor_ratio)
#             n_sliding = batch_size - n_anchor

#         # sliding 暂时也空 → 全部从 anchor 采
#         if len(self.sliding_storage) == 0:
#             n_anchor  = batch_size
#             n_sliding = 0

#         anchor_ind  = np.random.randint(0, len(self.anchor_storage),  size=n_anchor)  if n_anchor  > 0 else []
#         sliding_ind = np.random.randint(0, len(self.sliding_storage), size=n_sliding) if n_sliding > 0 else []

#         samples = [self.anchor_storage[i]  for i in anchor_ind] + \
#                   [self.sliding_storage[i] for i in sliding_ind]

#         if gdroweight is not None:
#             x, y, u, r, d, ret = [], [], [], [], [], []
#             for S in samples:
#                 X, Y, U, R, D, Ret = S
#                 x.append(np.array(X, copy=False))
#                 y.append(np.array(Y, copy=False))
#                 u.append(np.array(U, copy=False))
#                 r.append(np.array(R, copy=False))
#                 d.append(np.array(D, copy=False))
#                 ret.append(np.array(Ret, copy=False))
#             return (np.array(x), np.array(y), np.array(u),
#                     np.array(r).reshape(-1, 1),
#                     np.array(d).reshape(-1, 1),
#                     np.array(ret).reshape(-1, 1))
#         else:
#             x, y, u, r, d = [], [], [], [], []
#             for S in samples:
#                 X, Y, U, R, D = S
#                 x.append(np.array(X, copy=False))
#                 y.append(np.array(Y, copy=False))
#                 u.append(np.array(U, copy=False))
#                 r.append(np.array(R, copy=False))
#                 d.append(np.array(D, copy=False))
#             return (np.array(x), np.array(y), np.array(u),
#                     np.array(r).reshape(-1, 1),
#                     np.array(d).reshape(-1, 1))

#     def __len__(self):
#         return len(self.anchor_storage) + len(self.sliding_storage)
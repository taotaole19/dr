import torch
import torch.nn as nn
import pdb
from torch.distributions import Normal
from mpmodel.backbone_risklearner import build_mlp_icrpm, Decoder_me_icrpm, Encoder, MuSigmaEncoder, Decoder

# Define RiskLearner for the current task distribution and the current backbone parameters...
class RiskLearner(nn.Module):
    """
    Implements risklearner for functions of arbitrary dimensions.
    x_dim : int Dimension of x values.
    y_dim : int Dimension of y values.
    r_dim : int Dimension of output representation r.
    z_dim : int Dimension of latent variable z.
    h_dim : int Dimension of hidden layer in encoder and decoder.
    """

    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(RiskLearner, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Initialize networks
        self.xy_to_r = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim)

    def aggregate(self, r_i):
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        batch_size, num_points, _ = x.size()
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)

        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        return self.r_to_mu_sigma(r)

    def forward(self, x, y, output_type):
        """
        returns a distribution over target points y_target. We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        batch_size, num, x_dim = x.size()
        _, _, y_dim = y.size()

        if self.training:
            mu, sigma = self.xy_to_mu_sigma(x, y)
            z_variational_posterior = Normal(mu, sigma)
            z_sample = z_variational_posterior.rsample([1])

            p_y_pred = self.xz_to_y(x, z_sample, output_type)
            return p_y_pred, z_variational_posterior


class RiskLearner_icrpm(nn.Module):
    """
    Implements risklearner for functions of arbitrary dimensions.
    x_dim : int Dimension of x values.
    y_dim : int Dimension of y values.
    r_dim : int Dimension of output representation r.
    z_dim : int Dimension of latent variable z.
    h_dim : int Dimension of hidden layer in encoder and decoder.
    """

    def __init__(self, x_dim, y_dim, h_dim, d_model, emb_depth, nhead, dim_feedforward, dropout, num_layers):
        super(RiskLearner_icrpm, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        # self.r_dim = r_dim
        # self.z_dim = z_dim
        self.h_dim = h_dim
        self.d_model = d_model
        self.emb_depth = emb_depth

        self.xz_to_y_me = Decoder_me_icrpm(x_dim, d_model, h_dim, y_dim)

        # token embedding
        self.embedder = build_mlp_icrpm(x_dim + y_dim, d_model, d_model, emb_depth)
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def construct_input(self, last_risk_x, last_risk_y, risk_x, risk_y):
        x_y_ctx = torch.cat((last_risk_x, last_risk_y), dim=-1)
        # batch_size, num_ctx, dim
        if risk_y is None:
            fake_risk_y = torch.zeros(risk_x.shape[0], risk_x.shape[1], self.y_dim)
            fake_risk_y = fake_risk_y.to(risk_x.device)
            x_0_tar = torch.cat((risk_x, fake_risk_y), dim=-1)
        else:
            x_0_tar = torch.cat((risk_x, torch.zeros_like(risk_y)), dim=-1)
        # batch_size, num_tar, dim
        inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        # batch_size, 2*self.svpg.svpg_rollout_length*self.nagents, dim
        return inp

    def create_mask(self, last_risk_x, last_risk_y, risk_x, risk_y):
        num_ctx = last_risk_x.shape[1] #10*10
        num_tar = risk_x.shape[1] #10*10
        num_all = num_ctx + num_tar #200
        mask = torch.zeros(num_all, num_all, device='cuda').fill_(float('-inf'))
        mask[:, :num_ctx] = 0.0
        return mask, num_tar

    def aggregate(self, r_i):
        return torch.mean(r_i, dim=1)


    def forward(self, last_risk_x, last_risk_y, risk_x, risk_y, output_type):
        """
        returns a distribution over target points y_target. We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        # if len(x.size()) == 2:
        #     x = x.unsqueeze(-1)
        # batch_size, num, x_dim = risk_x.size()
        # _, _, y_dim = y.size()
        inp = self.construct_input(last_risk_x, last_risk_y, risk_x, risk_y) #(1, 200, nparams+1)
        mask, num_tar = self.create_mask(last_risk_x, last_risk_y, risk_x, risk_y) #mask:(200,200)
        embeddings = self.embedder(inp)
        full_out = self.encoder(embeddings, mask=mask)
        z_target = full_out[:, -num_tar:]
        if output_type == 'probabilistic':
            p_y_pred = self.xz_to_y_me(risk_x, z_target, output_type)
            return p_y_pred
        elif output_type == "deterministic":
            mu, sigma = self.xz_to_y_me(risk_x, z_target, output_type)
            return mu, sigma

        # if self.training:
        #     mu, sigma = self.xy_to_mu_sigma(x, y)
        #     z_variational_posterior = Normal(mu, sigma)
        #     z_sample = z_variational_posterior.rsample([1])

        #     p_y_pred = self.xz_to_y(x, z_sample, output_type)
        #     return p_y_pred, z_variational_posterior

class RiskLearner_np(nn.Module):
    """
    Neural Process 版 RiskLearner（MLP encoder + 均值聚合 + MLP decoder）。

    接口与 RiskLearner_icrpm 完全一致：
        forward(last_risk_x, last_risk_y, risk_x, risk_y, output_type)

    训练方式仍然是 icrpm 风格——由外部 trainer 显式提供 context (last_risk_x,
    last_risk_y) 和 target (risk_x, [risk_y])，模型在每次 forward 内
    重新对 context 进行编码并聚合，再解码 target。
    """

    def __init__(self, x_dim, y_dim, h_dim,
                 d_model, emb_depth,
                 nhead, dim_feedforward, dropout, num_layers):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.d_model = d_model        # 当作 NP 中的 r_dim 使用
        self.emb_depth = emb_depth
        # nhead / dim_feedforward / dropout / num_layers 仅为接口兼容，未使用

        # NP Encoder：(x_i, y_i) -> r_i
        self.xy_to_r = build_mlp_icrpm(
            dim_in=x_dim + y_dim,
            dim_hid=d_model,
            dim_out=d_model,
            depth=max(emb_depth, 2),
        )

        # NP Decoder：(x_target, r) -> Normal(mu, sigma)
        self.xz_to_y_me = Decoder_me_icrpm(x_dim, d_model, h_dim, y_dim)

    # ─────────────────────────────────────────────
    def aggregate(self, r_i):
        """对 context 维度做均值聚合，得到单个全局表征 r。"""
        return torch.mean(r_i, dim=1, keepdim=True)   # (B, 1, d_model)

    # ─────────────────────────────────────────────
    def forward(self, last_risk_x, last_risk_y, risk_x, risk_y, output_type):
        """
        last_risk_x : (B, num_ctx, x_dim)
        last_risk_y : (B, num_ctx, y_dim)
        risk_x      : (B, num_tar, x_dim)
        risk_y      : (B, num_tar, y_dim) 或 None  —— 仅为接口对齐，前向不使用
        output_type : "probabilistic" | "deterministic"
        """
        # 1) 编码每个 context (x_i, y_i) -> r_i
        ctx_pairs = torch.cat((last_risk_x, last_risk_y), dim=-1)   # (B, num_ctx, x_dim+y_dim)
        r_i = self.xy_to_r(ctx_pairs)                               # (B, num_ctx, d_model)

        # 2) 聚合为单一全局表征 r
        r = self.aggregate(r_i)                                     # (B, 1, d_model)

        # 3) 广播到每个 target 点
        num_tar = risk_x.shape[1]
        z_target = r.expand(-1, num_tar, -1).contiguous()           # (B, num_tar, d_model)

        # 4) 解码
        if output_type == "probabilistic":
            return self.xz_to_y_me(risk_x, z_target, output_type)   # Normal 分布
        elif output_type == "deterministic":
            mu, sigma = self.xz_to_y_me(risk_x, z_target, output_type)
            return mu, sigma
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

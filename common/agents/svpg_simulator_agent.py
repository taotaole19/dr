import gym
import numpy as np
import logging

import torch
from common.envs.randomized_vecenv import make_vec_envs

from common.svpg.svpg import SVPG

from common.utils.rollout_evaluation import evaluate_policy, check_solved, cvar_evaluate_policy, gdro_evaluate_policy
from common.agents.ddpg.replay_buffer import ReplayBuffer
from common.sampler.sampler import (MP_BatchSampler, Diverse_MP_BatchSampler, MP_BatchSampler_icrpm, Diverse_MP_BatchSampler_icrpm)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
import wandb
from scipy.stats import spearmanr

def _uses_sampler(algo: str) -> bool:
    """True only for the two algorithms that own a learned sampler."""
    # return algo in ('mpts', 'icrpm')
    return algo in ('mpts', 'icrpm') or 'diverse' in algo   # ← 加上 diverse 变体


def _uses_agent_policy_scores(algo: str) -> bool:
    """True for baselines that score candidate tasks using the agent policy gradients."""
    return algo in ('dats', 'tdps')

class SVPGSimulatorAgent(object):
    """Simulation object which creates randomized environments based on specified params, 
    handles SVPG-based policy search to create envs, 
    and evaluates controller policies in those environments
    """

    def __init__(self,
                 reference_env_id,
                 randomized_env_id,
                 randomized_eval_env_id,
                 agent_name,
                 nagents,
                 nparams,
                 temperature,
                 svpg_rollout_length,
                 svpg_horizon,
                 max_step_length,
                 reward_scale,
                 initial_svpg_steps,
                 max_env_timesteps,
                 episodes_per_instance,
                 discrete_svpg,
                 load_discriminator,
                 freeze_discriminator,
                 freeze_agent,
                 seed,
                 args,
                 train_svpg=True,
                 particle_path="",
                 discriminator_batchsz=320,
                 randomized_eval_episodes=3,
                 ):

        assert nagents > 2

        self.reference_env_id = reference_env_id
        self.randomized_env_id = randomized_env_id
        self.randomized_eval_env_id = randomized_eval_env_id
        self.agent_name = agent_name

        self.log_distances = reference_env_id.find('Lunar') == -1

        self.randomized_eval_episodes = randomized_eval_episodes
        if reference_env_id.find('Pusher') != -1:
            self.randomized_eval_episodes = 10
        elif reference_env_id.find('Lunar') != -1:
            self.randomized_eval_episodes = 10
        elif reference_env_id.find('Ergo') != -1:
            self.randomized_eval_episodes = 10

        self.reference_env = make_vec_envs(reference_env_id, seed, nagents)
        self.randomized_env = make_vec_envs(randomized_env_id, seed, nagents)
        self.randomized_eval_env = make_vec_envs(randomized_eval_env_id, seed, nagents)

        self.state_dim = self.reference_env.observation_space.shape[0]
        self.action_dim = self.reference_env.action_space.shape[0]
    
        if reference_env_id.find('Pusher') != -1:
            self.hard_env = make_vec_envs('Pusher3DOFHard-v0', seed, nagents)
        elif reference_env_id.find('Lunar') != -1:
            self.hard_env = make_vec_envs('LunarLander10-v0', seed, nagents)
        elif reference_env_id.find('Backlash') != -1:
            self.hard_env = make_vec_envs('ErgoReacherRandomizedBacklashHard-v0', seed, nagents)
        else:
            self.hard_env = make_vec_envs('ErgoReacher4DOFRandomizedHard-v0', seed, nagents)

        self.sampled_regions = [[] for _ in range(nparams)]

        self.nagents = nagents
        self.nparams = self.randomized_env.randomization_space.shape[0]
        assert self.nparams == nparams, "Double check number of parameters: Args: {}, Env: {}".format(
            nparams, self.nparams)

        self.svpg_horizon = svpg_horizon
        self.initial_svpg_steps = initial_svpg_steps
        self.max_env_timesteps = max_env_timesteps
        self.episodes_per_instance = episodes_per_instance
        self.discrete_svpg = discrete_svpg

        self.freeze_discriminator = freeze_discriminator
        self.freeze_agent = freeze_agent

        self.train_svpg = train_svpg
        
        self.agent_eval_frequency = max_env_timesteps * nagents 
        if self.log_distances:
            self.agent_eval_frequency = 25000
        else:
            self.agent_eval_frequency = 50000

        self.seed = seed
        self.svpg_timesteps = 0
        self.agent_timesteps = 0
        self.agent_timesteps_since_eval = 0
        
        from mpmodel.risklearner import RiskLearner, RiskLearner_icrpm, RiskLearner_np
        from mpmodel.new_trainer_risklearner import RiskLearnerTrainer, RiskLearnerTrainer_Markov
        self.args = args

        # ── sampler: only created for mpts / icrpm ────────────────────────
        self.sampler = None   # sentinel; checked via _uses_sampler() throughout
        # pdts新增:
        # ── pdts → ps_diverse_mpts 别名（与第一版保持一致）─────────────────
        if self.args.algo == 'pdts':
            self.args.algo = 'ps_diverse_icrpm'

        if self.args.algo == 'vae_pdts':
            self.args.algo = 'ps_diverse_mpts'
        
        # ── diverse / posterior-sampling 标志位 ──────────────────────────
        if 'ps' in self.args.algo:
            self.args.posterior_sampling = True
        diversity_type = 'msdmin' if 'diverse' in self.args.algo else None
        self.diversity_type = diversity_type   # 供 select_action 引用

        if self.args.algo == 'mpts' or 'mpts' in self.args.algo:
            self.risklearner = RiskLearner(self.nparams, 1, 10, 10, 10).to(device)
            self.risklearner_optimizer = torch.optim.Adam(
                self.risklearner.parameters(), lr=args.sampler_lr
            )
            self.risklearner_trainer = RiskLearnerTrainer(
                device, self.risklearner, self.risklearner_optimizer,
                output_type=args.output_type, kl_weight=args.kl_weight,
                posterior_sampling=args.posterior_sampling,
                diversity_type=diversity_type,
            )
            if 'diverse' in args.algo:
                self.sampler = Diverse_MP_BatchSampler(args, self.risklearner_trainer, 
                                    args.sampling_gamma_0,
                                    args.sampling_gamma_1,
                                    args.sampling_gamma_2,)
            else:
                self.sampler = MP_BatchSampler(args, self.risklearner_trainer, 
                                    args.sampling_gamma_0,
                                    args.sampling_gamma_1,)
                                    
        elif self.args.algo == 'icrpm' or 'icrpm' in self.args.algo:
            self.risklearner = RiskLearner_icrpm(
                x_dim=self.nparams, y_dim=1, h_dim=10,
                d_model=64, emb_depth=4, nhead=4, dim_feedforward=128,
                dropout=0.0, num_layers=2
            ).to(device)
            # self.risklearner = RiskLearner_np(
            #     x_dim=self.nparams, y_dim=1, h_dim=10,
            #     d_model=64, emb_depth=4, nhead=4, dim_feedforward=128,
            #     dropout=0.0, num_layers=2
            # ).to(device)
            self.risklearner_optimizer = torch.optim.Adam(
                self.risklearner.parameters(), lr=args.sampler_lr
            )
            self.risklearner_trainer = RiskLearnerTrainer_Markov(
                device, self.risklearner, self.risklearner_optimizer,
                diversity_type=diversity_type,
                posterior_sampling=getattr(args, 'posterior_sampling', False),
            )
            if diversity_type is not None:
                self.sampler = Diverse_MP_BatchSampler_icrpm(
                    args, self.risklearner_trainer,
                    args.sampling_gamma_0,
                    args.sampling_gamma_1,
                    args.sampling_gamma_2,
                )
            else:
                self.sampler = MP_BatchSampler_icrpm(
                    args, self.risklearner_trainer,
                    args.sampling_gamma_0, args.sampling_gamma_1,
                )

        if not self.freeze_agent:
            self.replay_buffer = ReplayBuffer()
        else:
            self.replay_buffer = None

        self.svpg = SVPG(nagents=nagents,
                         nparams=self.nparams,
                         max_step_length=max_step_length,
                         svpg_rollout_length=svpg_rollout_length,
                         svpg_horizon=svpg_horizon,
                         temperature=temperature,
                         discrete=self.discrete_svpg,
                         kld_coefficient=0.0)

        if particle_path != "":
            logger.info("Loading particles from: {}".format(particle_path))
            self.svpg.load(directory=particle_path)

        # ── ohtm: task loss history (keyed by rounded param tuple) ───────
        self.task_loss_history = {}   # {param_tuple: float}

    # ─────────────────────────────────────────────────────────────────────────
    # Baseline helpers: DATS, TDPS, OHTM
    # Inputs are *normalised* parameter vectors in [0,1]^nparams produced by
    # the SVPG / uniform sampler, and an agent_policy whose gradient signals
    # are used to score tasks.
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_dats_scores(self, candidate_params: np.ndarray,
                             agent_policy, eta: float = 1.0) -> torch.Tensor:
        """
        DATS: score = softmax( eta * <g_support_i, mean(g_query) > )
        candidate_params: (N, nparams) numpy array in [0,1]
        Returns: (N,) tensor of softmax-normalised scores.
        """
        support_grads = []
        query_grads   = []

        for params in candidate_params:
            # randomize env with this single param vector (nagents copies)
            self.randomized_env.randomize(
                randomized_values=np.tile(params, (self.nagents, 1))
            )
            # collect two independent rollouts to mimic support / query split
            _, ret_s, _ = self.rollout_agent(agent_policy, reference=False)
            _, ret_q, _ = self.rollout_agent(agent_policy, reference=False)

            # use scalar return as proxy loss; gradients w.r.t. policy params
            loss_s = -torch.tensor(float(np.mean(ret_s)), requires_grad=False)
            loss_q = -torch.tensor(float(np.mean(ret_q)), requires_grad=False)

            # policy gradient approximation via log-return proxy
            # For a gradient-free approximation we use the negated returns as
            # flat "gradient" vectors (one scalar each), which degenerates to
            # the inner-product being return_s * mean(return_q).
            support_grads.append(torch.tensor([float(np.mean(ret_s))]))
            query_grads.append(torch.tensor([float(np.mean(ret_q))]))

        support_grads = torch.stack(support_grads)   # (N, 1)
        query_grads   = torch.stack(query_grads)     # (N, 1)
        avg_query     = query_grads.mean(dim=0)      # (1,)
        inner         = (support_grads * avg_query).sum(dim=1)  # (N,)
        return torch.softmax(eta * inner, dim=0)

    def _compute_tdps_scores(self, candidate_params: np.ndarray,
                             agent_policy) -> torch.Tensor:
        """
        TDPS: difficulty = |return_support - return_query|
        Higher inconsistency → harder task → higher score.
        candidate_params: (N, nparams) numpy array in [0,1]
        Returns: (N,) tensor of raw inconsistency scores (not normalised).
        """
        scores = []
        for params in candidate_params:
            self.randomized_env.randomize(
                randomized_values=np.tile(params, (self.nagents, 1))
            )
            _, ret_s, _ = self.rollout_agent(agent_policy, reference=False)
            _, ret_q, _ = self.rollout_agent(agent_policy, reference=False)
            inconsistency = abs(float(np.mean(ret_s)) - float(np.mean(ret_q)))
            scores.append(inconsistency)
        return torch.tensor(scores)

    def _ohtm_select(self, candidate_params: np.ndarray,
                     num_select: int) -> np.ndarray:
        """
        OHTM: mix top-half hardest seen tasks with random unseen tasks.
        candidate_params: (N, nparams) numpy array in [0,1]
        Returns: (num_select, nparams) selected parameter array.
        """
        N = len(candidate_params)
        candidate_keys = [
            tuple(np.round(p, 4)) for p in candidate_params
        ]

        num_hard = num_select // 2
        seen_indices = [
            i for i, key in enumerate(candidate_keys)
            if key in self.task_loss_history
        ]

        hard_idx = []
        if seen_indices:
            seen_indices.sort(
                key=lambda i: self.task_loss_history[candidate_keys[i]],
                reverse=True,   # highest loss = hardest
            )
            hard_idx = seen_indices[:num_hard]

        num_random = num_select - len(hard_idx)
        remaining  = [i for i in range(N) if i not in hard_idx]
        rand_idx   = np.random.choice(remaining, size=num_random,
                                      replace=False).tolist()

        selected_idx    = hard_idx + rand_idx
        selected_params = candidate_params[selected_idx]
        return selected_params

    def _ohtm_update_history(self, params: np.ndarray,
                             returns: np.ndarray) -> None:
        """Store per-task losses (negated returns) for future hard-mining."""
        for p, r in zip(params, returns):
            key = tuple(np.round(p, 4))
            self.task_loss_history[key] = -float(r)   # higher loss = lower return

    def select_action(self, agent_policy):
        """Select an action based on SVPG policy, where an action is the delta in each dimension."""

        # total tasks needed for one SVPG step
        _total_tasks = self.nagents * self.svpg.svpg_rollout_length
        _shape_3d    = (self.nagents, self.svpg.svpg_rollout_length, self.svpg.nparams)

        if self.agent_timesteps >= self.args.uniform_sample_steps * self.args.max_agent_timesteps:
            algo = self.args.algo

            if _uses_sampler(algo):
                if self.diversity_type is not None:
                    # diverse 路径（mpts diverse / icrpm diverse 共用）
                    (simulation_instances,
                     diversified_score,
                     combine_local_diverse_score,
                     combine_local_acquisition_score) = self.sampler.sample_tasks(
                        shape=_shape_3d,
                        multiplier=self.args.sampler_multiplier,
                    )
                else:
                    # 普通路径（mpts / icrpm 非 diverse）
                    simulation_instances = self.sampler.sample_tasks(
                        shape=_shape_3d,
                        multiplier=self.args.sampler_multiplier,
                    )

            elif algo == 'dats':
                # ── DATS: gradient inner-product weighting ────────────────
                # Score a larger candidate pool, then pick top-_total_tasks.
                multiplier    = getattr(self.args, 'sampler_multiplier', 2)
                n_candidates  = int(multiplier * _total_tasks)
                candidates    = np.random.uniform(0.0, 1.0,
                                    size=(n_candidates, self.svpg.nparams))
                scores = self._compute_dats_scores(
                    candidates, agent_policy,
                    eta=getattr(self.args, 'dats_eta', 1.0),
                )
                _, top_idx = torch.topk(scores, k=_total_tasks)
                selected   = candidates[top_idx.cpu().numpy()]          # (_total_tasks, nparams)
                simulation_instances = selected.reshape(_shape_3d)

            elif algo == 'tdps':
                # ── TDPS: gradient inconsistency hard-task mining ─────────
                multiplier   = getattr(self.args, 'sampler_multiplier', 2)
                n_candidates = int(multiplier * _total_tasks)
                candidates   = np.random.uniform(0.0, 1.0,
                                    size=(n_candidates, self.svpg.nparams))
                scores = self._compute_tdps_scores(candidates, agent_policy)
                _, top_idx = torch.topk(scores, k=_total_tasks)
                selected   = candidates[top_idx.cpu().numpy()]
                simulation_instances = selected.reshape(_shape_3d)

            elif algo == 'ohtm':
                # ── OHTM: online hard-task mining with history ────────────
                multiplier   = getattr(self.args, 'sampler_multiplier', 2)
                n_candidates = int(multiplier * _total_tasks)
                candidates   = np.random.uniform(0.0, 1.0,
                                    size=(n_candidates, self.svpg.nparams))
                selected = self._ohtm_select(candidates, num_select=_total_tasks)
                simulation_instances = selected.reshape(_shape_3d)

            else:
                # ── all other algos: uniform random ───────────────────────
                simulation_instances = np.random.uniform(
                    0.0, 1.0, size=_shape_3d,
                )
        else:
            # warm-up phase: uniform random for every algo
            simulation_instances = np.random.uniform(
                0.0, 1.0, size=_shape_3d,
            )

        assert (self.nagents, self.svpg.svpg_rollout_length, self.svpg.nparams) == simulation_instances.shape

        randomized_returns = []
        randomized_dists = []

        simulation_instances = np.transpose(simulation_instances, (1, 0, 2))

        for t in range(self.svpg.svpg_rollout_length):
            logging.info('Iteration t: {}/{}'.format(t, self.svpg.svpg_rollout_length))

            self.randomized_env.randomize(randomized_values=simulation_instances[t])
            randomized_trajectory, randomized_return, randomized_dist = self.rollout_agent(
                agent_policy,
                reference=False,
                cvar=self.args.cvar if self.args.algo == 'drm' else None,
                gdroweight=self.args.gdroweight if self.args.algo == 'gdrm' else None,
            )

            randomized_returns.append(randomized_return)
            randomized_dists.append(randomized_dist)

            for i in range(self.nagents if self.args.algo != 'drm' else int((1 - self.args.cvar) * self.nagents)):
                self.agent_timesteps += len(randomized_trajectory[i])
                self.agent_timesteps_since_eval += len(randomized_trajectory[i])

        # ── initialise sampler-related variables used later ───────────────
        # These are only meaningful for mpts/icrpm but are referenced in the
        # info dict and wandb.log below, so we define safe defaults here.
        acquisition_score    = None
        acquisition_mean     = None
        acquisition_std      = None
        sampler_loss         = 0
        recon_loss           = 0
        kl_loss              = 0

        if self.svpg_timesteps >= self.initial_svpg_steps:
            if self.train_svpg:
                randomized_returns = np.array(randomized_returns)
                if self.args.use_dist:
                    randomized_returns = -np.array(randomized_dists)

                randomized_returns_flatten = torch.FloatTensor(
                    randomized_returns.reshape(self.svpg.svpg_rollout_length * self.nagents)
                ).to(device)
                simulation_instances_flatten = torch.FloatTensor(
                    simulation_instances.reshape(self.svpg.svpg_rollout_length * self.nagents, self.svpg.nparams)
                ).to(device)

                # ── sampler training: only for mpts / icrpm ────────────────
                if _uses_sampler(self.args.algo):
                    acquisition_score, acquisition_mean, acquisition_std = \
                        self.sampler.get_acquisition_score(simulation_instances_flatten)

                    randomized_returns_flatten_neg = -randomized_returns_flatten
                    if not self.args.no_batch_norm:
                        y = (randomized_returns_flatten_neg - randomized_returns_flatten_neg.mean()) \
                            / (randomized_returns_flatten_neg.std() + 1e-8)
                    else:
                        y = randomized_returns_flatten_neg

                    for i in range(self.args.sampler_train_times):
                        sampler_loss, recon_loss, kl_loss = self.sampler.train(
                            simulation_instances_flatten, y, i
                        )
                # ─────────────────────────────────────────────────────────

                # ── OHTM: update task loss history after rollouts ─────────
                if self.args.algo == 'ohtm':
                    # simulation_instances is (rollout_len, nagents, nparams)
                    # randomized_returns is (rollout_len, nagents)
                    params_flat  = simulation_instances.reshape(-1, self.nparams)
                    returns_flat = np.array(randomized_returns).reshape(-1)
                    self._ohtm_update_history(params_flat, returns_flat)
                # ─────────────────────────────────────────────────────────

            for dimension in range(self.nparams):
                self.sampled_regions[dimension] = np.concatenate([
                    self.sampled_regions[dimension],
                    simulation_instances[:, :, dimension].flatten(),
                ])

        solved_reference = info = None
        if self.agent_timesteps_since_eval >= self.agent_eval_frequency:
            self.agent_timesteps_since_eval %= self.agent_eval_frequency
            logger.info("Evaluating for {} episodes after timesteps: {} (SVPG), {} (Agent)".format(
                self.randomized_eval_episodes * self.nagents, self.svpg_timesteps, self.agent_timesteps))

            agent_reference_eval_rewards  = []
            agent_hard_eval_rewards       = []
            agent_randomized_eval_rewards = []
            avg_agent_randomized_eval_rewards = []

            final_dist_ref  = []
            final_dist_hard = []
            final_dist_rand = []
            avg_final_dist_rand = []
            eval_acquisition_scores = []

            rewards_ref, dist_ref = evaluate_policy(
                nagents=self.nagents, env=self.reference_env,
                agent_policy=agent_policy, replay_buffer=None,
                eval_episodes=5, max_steps=self.max_env_timesteps,
                return_rewards=True, add_noise=False,
                log_distances=self.log_distances,
            )
            rewards_hard, dist_hard = evaluate_policy(
                nagents=self.nagents, env=self.hard_env,
                agent_policy=agent_policy, replay_buffer=None,
                eval_episodes=5, max_steps=self.max_env_timesteps,
                return_rewards=True, add_noise=False,
                log_distances=self.log_distances,
            )
            agent_reference_eval_rewards += list(rewards_ref)
            agent_hard_eval_rewards      += list(rewards_hard)
            final_dist_ref  += [dist_ref]
            final_dist_hard += [dist_hard]

            for _ in range(self.randomized_eval_episodes):
                full_random_settings = np.random.uniform(0.0, 1.0, size=(self.nagents, self.nparams))
                self.randomized_eval_env.randomize(randomized_values=full_random_settings)
            
                # ── sampler acquisition at eval: only for mpts / icrpm ─────
                if _uses_sampler(self.args.algo):
                    full_random_settings_tensor = torch.FloatTensor(full_random_settings).to(device)
                    eval_acquisition_score, _, _ = self.sampler.get_acquisition_score(
                            full_random_settings_tensor
                            )
                    eval_acquisition_scores += list(
                        eval_acquisition_score.squeeze().cpu().detach().numpy()
                    )
                # ─────────────────────────────────────────────────────────

                rewards_rand, dist_rand = evaluate_policy(
                    nagents=self.nagents, env=self.randomized_eval_env,
                    agent_policy=agent_policy, replay_buffer=None,
                    eval_episodes=5, max_steps=self.max_env_timesteps,
                    return_rewards=True, add_noise=False,
                    log_distances=self.log_distances,
                )
                avg_rewards_rand = np.mean(rewards_rand.reshape(5, -1), axis=0)
                avg_dist_rand    = np.mean(dist_rand.reshape(5, -1), axis=0)

                agent_randomized_eval_rewards     += list(rewards_rand)
                final_dist_rand                   += [dist_rand]
                avg_agent_randomized_eval_rewards += list(avg_rewards_rand)
                avg_final_dist_rand               += [avg_dist_rand]

            evaluation_criteria_reference  = agent_reference_eval_rewards
            evaluation_criteria_randomized = agent_randomized_eval_rewards
            if self.log_distances:
                evaluation_criteria_reference  = final_dist_ref
                evaluation_criteria_randomized = final_dist_rand

            solved_reference  = check_solved(self.reference_env_id,       evaluation_criteria_reference)
            solved_randomized = check_solved(self.randomized_eval_env_id,  evaluation_criteria_randomized)

            # ── build info dict ───────────────────────────────────────────
            info = {
                'solved':            str(solved_reference),
                'solved_randomized': str(solved_randomized),
                'svpg_steps':        self.svpg_timesteps,
                'agent_timesteps':   self.agent_timesteps,
                # 'final_dist_ref_mean':    np.mean(final_dist_ref),
                # 'final_dist_ref_std':     np.std(final_dist_ref),
                # 'final_dist_ref_median':  np.median(final_dist_ref),
                # 'final_dist_hard_mean':   np.mean(final_dist_hard),
                # 'final_dist_hard_std':    np.std(final_dist_hard),
                # 'final_dist_hard_median': np.median(final_dist_hard),
                # 'final_dist_rand_mean':   np.mean(final_dist_rand),
                # 'final_dist_rand_std':    np.std(final_dist_rand),
                # 'final_dist_rand_median': np.median(final_dist_rand),
                'agent_reference_eval_rewards_mean':   np.mean(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_std':    np.std(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_median': np.median(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_min':    np.min(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_max':    np.max(agent_reference_eval_rewards),
                'agent_hard_eval_rewards_median':      np.median(agent_hard_eval_rewards),
                'agent_hard_eval_rewards_mean':        np.mean(agent_hard_eval_rewards),
                'agent_hard_eval_rewards_std':         np.std(agent_hard_eval_rewards),
                'agent_randomized_eval_rewards_mean':   np.mean(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_std':    np.std(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_median': np.median(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_min':    np.min(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_max':    np.max(agent_randomized_eval_rewards),
                'sampler_loss': sampler_loss,
                'recon_loss':   recon_loss,
                'kl_loss':      kl_loss,
            }


            # ── CVaR slices (always logged) ───────────────────────────────
            cvar50 = np.sort(avg_agent_randomized_eval_rewards)[:int(len(avg_agent_randomized_eval_rewards) * 0.50)]
            cvar30 = np.sort(avg_agent_randomized_eval_rewards)[:int(len(avg_agent_randomized_eval_rewards) * 0.30)]
            cvar10 = np.sort(avg_agent_randomized_eval_rewards)[:int(len(avg_agent_randomized_eval_rewards) * 0.10)]
            cvar5  = np.sort(avg_agent_randomized_eval_rewards)[:int(len(avg_agent_randomized_eval_rewards) * 0.05)]

            wandb_log = {
                'eval_hard_rewards':        np.mean(agent_hard_eval_rewards),
                'eval/reference_rewards':   np.mean(agent_reference_eval_rewards),
                'eval/unif_rewards':        np.mean(agent_randomized_eval_rewards),
                'eval/cvar50_rewards':      np.mean(cvar50),
                'eval/cvar30_rewards':      np.mean(cvar30),
                'eval/cvar10_rewards':      np.mean(cvar10),
                'eval/cvar5_rewards':       np.mean(cvar5),
                'train/svpg_timesteps':     self.svpg_timesteps,
                'step': self.agent_timesteps,
            }

            # ── sampler-only wandb fields ─────────────────────────────────
            if _uses_sampler(self.args.algo) and acquisition_score is not None:
                acq_np  = acquisition_score.squeeze().cpu().detach().numpy()
                ret_np  = randomized_returns_flatten.squeeze().cpu().detach().numpy()

            wandb.log(wandb_log)

        self.svpg_timesteps += 1
        return solved_reference, info

    def rollout_agent(self, agent_policy, reference=True, eval_episodes=None, cvar=None, gdroweight=None):
        if cvar:
            trajectory, avg_returns, avg_dists, env_steps = cvar_evaluate_policy(
                nagents=self.nagents, env=self.randomized_env,
                agent_policy=agent_policy, replay_buffer=self.replay_buffer,
                eval_episodes=self.episodes_per_instance, max_steps=self.max_env_timesteps,
                freeze_agent=self.freeze_agent, add_noise=True,
                log_distances=self.log_distances, cvar=cvar,
            )
            return trajectory, avg_returns, avg_dists
        elif gdroweight:
            trajectory, avg_returns, avg_dists = gdro_evaluate_policy(
                nagents=self.nagents, env=self.randomized_env,
                agent_policy=agent_policy, replay_buffer=self.replay_buffer,
                eval_episodes=self.episodes_per_instance, max_steps=self.max_env_timesteps,
                freeze_agent=self.freeze_agent, add_noise=True,
                log_distances=self.log_distances, gdroweight=gdroweight,
            )
            return trajectory, avg_returns, avg_dists
        else:
            if reference:
                if eval_episodes is None:
                    eval_episodes = self.episodes_per_instance
                trajectory, avg_returns, avg_dists = evaluate_policy(
                    nagents=self.nagents, env=self.reference_env,
                    agent_policy=agent_policy, replay_buffer=None,
                    eval_episodes=eval_episodes, max_steps=self.max_env_timesteps,
                    freeze_agent=True, add_noise=False,
                    log_distances=self.log_distances,
                )
            else:
                trajectory, avg_returns, avg_dists = evaluate_policy(
                    nagents=self.nagents, env=self.randomized_env,
                    agent_policy=agent_policy, replay_buffer=self.replay_buffer,
                    eval_episodes=self.episodes_per_instance, max_steps=self.max_env_timesteps,
                    freeze_agent=self.freeze_agent, add_noise=True,
                    log_distances=self.log_distances,
                )
            return trajectory, avg_returns, avg_dists

    def sample_trajectories(self, batch_size):
        indices = np.random.randint(0, len(self.extracted_trajectories['states']), batch_size)
        states      = self.extracted_trajectories['states']
        actions     = self.extracted_trajectories['actions']
        next_states = self.extracted_trajectories['next_states']
        trajectories = []
        for i in indices:
            trajectories.append(np.concatenate([
                np.array(states[i]),
                np.array(actions[i]),
                np.array(next_states[i]),
            ], axis=-1))
        return trajectories

    def evaluate_in_full_range(self, agent_policy):
        assert self.reference_env_id.find('Lunar') != -1
        self.full_eval_env = make_vec_envs('LunarLanderRandomizedFull-v0', self.seed, self.nagents)
        rewards_grids = []
        dist_grids = []
        for tau in np.linspace(0.0, 1.0, 23):
            full_settings = np.ones((self.nagents, self.nparams)) * tau
            self.full_eval_env.randomize(randomized_values=full_settings)
            rewards_grid, dist_grid = evaluate_policy(
                nagents=self.nagents, env=self.full_eval_env,
                agent_policy=agent_policy, replay_buffer=None,
                eval_episodes=5, max_steps=self.max_env_timesteps,
                return_rewards=True, add_noise=False,
                log_distances=self.log_distances,
            )
            tau_scaled = self.full_eval_env.rescale(0, tau)
            wandb.log({
                'test_full_range/rewards': np.mean(rewards_grid),
                'tau_scaled': int(round(tau_scaled)),
            })
            rewards_grids.append(np.mean(rewards_grid))
        full_tau_scaled = self.full_eval_env.rescale(0, np.linspace(0.0, 1.0, 23))
        return np.array(rewards_grids), full_tau_scaled

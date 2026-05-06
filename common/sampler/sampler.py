import multiprocessing as mp

import gym
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MP_BatchSampler(object):
    def __init__(self, args,risk_learner_trainer, gamma_0, gamma_1):
        self.risklearner_trainer = risk_learner_trainer
        self.args = args
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.current_epoch = 0

    def get_acquisition_score(self, tasks):
        acquisition_score, acquisition_mean, acquisition_std = self.risklearner_trainer.acquisition_function(tasks, self.gamma_0, self.gamma_1)
        return acquisition_score, acquisition_mean, acquisition_std

    def sample_tasks(self, shape, multiplier, init_dist='Uniform', test=False):
        candidate_tasks = torch.rand(int(multiplier*shape[0]*shape[1]),shape[2])
        # candidate_tasks = np.random.uniform(0.0, 1.0, size=(int(multiplier*shape[0]*shape[1]),shape[2]))
        acquisition_score, acquisition_mean, acquisition_std = self.get_acquisition_score(candidate_tasks) # candidate tasks 15 * loss 1
        acquisition_score = acquisition_score.squeeze(1) # candidate tasks 15
        if not self.args.no_add_random:
            selected_values, selected_index = torch.topk(acquisition_score, k=shape[0]*shape[1]//2)
        else:
            selected_values, selected_index = torch.topk(acquisition_score, k=shape[0]*shape[1])
        mask = ~torch.isin(torch.arange(0, int(multiplier*shape[0]*shape[1])), selected_index.cpu())
        unselected_index = torch.arange(0, int(multiplier*shape[0]*shape[1]))[mask]
        index=torch.cat((selected_index.cpu(),unselected_index),dim=0)[:shape[0]*shape[1]][torch.randperm(shape[0]*shape[1])] # num_tasks 10
        index = index.cpu()
        tasks = candidate_tasks[index]
        tasks = tasks.view(shape[0],shape[1],shape[2]).numpy()

        return tasks
    
    def train(self, tasks, y, i):
        loss, recon_loss, kl_loss = self.risklearner_trainer.train(tasks, y)
        return loss, recon_loss, kl_loss
    
class Diverse_MP_BatchSampler(MP_BatchSampler):
    def __init__(self, args,risk_learner_trainer, gamma_0, gamma_1, gamma_2):
        self.gamma_2 = gamma_2
        super(Diverse_MP_BatchSampler, self).__init__(args,risk_learner_trainer, gamma_0, gamma_1)

    def get_acquisition_score(self, tasks, real_batch_size=None, diversified=False):
        if real_batch_size is None:
            real_batch_size = int(tasks.shape[0])
        if diversified:
            best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score = self.risklearner_trainer.acquisition_function(tasks,  self.gamma_0, self.gamma_1, self.gamma_2, real_batch_size=real_batch_size)
            return best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score
        else:
            acquisition_score, acquisition_mean, acquisition_std = self.risklearner_trainer.acquisition_function(tasks, self.gamma_0, self.gamma_1, self.gamma_2, pure_acquisition=True, real_batch_size=real_batch_size)
            return acquisition_score, acquisition_mean, acquisition_std


    def sample_tasks(self, shape, multiplier, init_dist='Uniform', test=False):

        candidate_tasks = torch.rand(int(multiplier*shape[0]*shape[1]),shape[2])
        # candidate_tasks = np.random.uniform(0.0, 1.0, size=(int(multiplier*shape[0]*shape[1]),shape[2]))
        best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score = self.get_acquisition_score(candidate_tasks, real_batch_size=int(shape[0]*shape[1]), diversified=True) # candidate tasks 15 * loss 1
        index = best_batch_id
        tasks = candidate_tasks[index]
        tasks = tasks.view(shape[0],shape[1],shape[2]).numpy()

        return tasks, diversified_score, combine_local_diverse_score, combine_local_acquisition_score


class MP_BatchSampler_icrpm(object):
    def __init__(self, args, risk_learner_trainer, gamma_0, gamma_1):
        self.risklearner_trainer = risk_learner_trainer  # RiskLearnerTrainer_icrpm instance
        self.args = args
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.current_epoch = 0

    def get_acquisition_score(self, tasks):
        # RiskLearnerTrainer_icrpm.acquisition_function returns
        # (acquisition_score, output_mu, output_sigma)  — same signature as before.
        # When history is empty it returns random scores, so no special case needed here.
        acquisition_score, output_mu, output_sigma = \
            self.risklearner_trainer.acquisition_function(tasks, self.gamma_0, self.gamma_1)
        return acquisition_score, output_mu, output_sigma

    def sample_tasks(self, shape, multiplier, init_dist='Uniform', test=False):
        candidate_tasks = torch.rand(
            int(multiplier * shape[0] * shape[1]), shape[2]
        )  # (multiplier * num_tasks, n_params)

        acquisition_score, acquisition_mean, acquisition_std = \
            self.get_acquisition_score(candidate_tasks)
        acquisition_score = acquisition_score.squeeze(1)  # (num_candidates,)

        if not self.args.no_add_random:
            selected_values, selected_index = torch.topk(
                acquisition_score, k=shape[0] * shape[1] // 4
            )
        else:
            selected_values, selected_index = torch.topk(
                acquisition_score, k=shape[0] * shape[1]
            )

        mask = ~torch.isin(
            torch.arange(0, int(multiplier * shape[0] * shape[1])),
            selected_index.cpu()
        )
        unselected_index = torch.arange(
            0, int(multiplier * shape[0] * shape[1])
        )[mask]

        index = torch.cat(
            (selected_index.cpu(), unselected_index), dim=0
        )[: shape[0] * shape[1]][torch.randperm(shape[0] * shape[1])]
        index = index.cpu()

        tasks = candidate_tasks[index]
        tasks = tasks.view(shape[0], shape[1], shape[2]).numpy()
        return tasks

    def train(self, tasks, y, i=None):
        """
        RiskLearnerTrainer_icrpm.train() returns either:
          - a plain scalar  loss.item()          (cold-start, first step)
          - a tuple         (total_loss.item(), 0, 0)   (normal steps)

        We normalise both cases to always return (loss, recon_loss, kl_loss)
        so the calling code can stay the same as before.
        """
        result = self.risklearner_trainer.train(tasks, y, i)

        if isinstance(result, tuple):
            # normal case: (total_loss, 0, 0)
            loss, recon_loss, kl_loss = result
        else:
            # cold-start case: bare scalar
            loss, recon_loss, kl_loss = result, 0, 0

        return loss, recon_loss, kl_loss


class Diverse_MP_BatchSampler_icrpm(MP_BatchSampler_icrpm):
    """
    Diverse 版本的 icrpm 采样器，与 Diverse_MP_BatchSampler 完全对称。
    RiskLearnerTrainer_Markov 需以 diversity_type='msd' 初始化。
    """
    def __init__(self, args, risk_learner_trainer, gamma_0, gamma_1, gamma_2):
        self.gamma_2 = gamma_2
        super().__init__(args, risk_learner_trainer, gamma_0, gamma_1)

    def get_acquisition_score(self, tasks, real_batch_size=None, diversified=False):
        if real_batch_size is None:
            real_batch_size = int(tasks.shape[0])
        if diversified:
            # 返回 5 元组：best_batch_id, diversified_score,
            #              combine_local_diverse_score, combine_local_acquisition_score,
            #              acquisition_score
            (best_batch_id,
             diversified_score,
             combine_local_diverse_score,
             combine_local_acquisition_score,
             acquisition_score) = self.risklearner_trainer.acquisition_function(
                tasks,
                self.gamma_0, self.gamma_1, self.gamma_2,
                real_batch_size=real_batch_size,
            )
            return (best_batch_id, diversified_score,
                    combine_local_diverse_score, combine_local_acquisition_score,
                    acquisition_score)
        else:
            # pure_acquisition=True：仅返回 3 元组，供 get_acquisition_score 外部调用
            acquisition_score, acquisition_mean, acquisition_std = \
                self.risklearner_trainer.acquisition_function(
                    tasks,
                    self.gamma_0, self.gamma_1, self.gamma_2,
                    pure_acquisition=True,
                    real_batch_size=real_batch_size,
                )
            return acquisition_score, acquisition_mean, acquisition_std

    def sample_tasks(self, shape, multiplier, init_dist='Uniform', test=False):
        candidate_tasks = torch.rand(
            int(multiplier * shape[0] * shape[1]), shape[2]
        )

        (best_batch_id,
         diversified_score,
         combine_local_diverse_score,
         combine_local_acquisition_score,
         acquisition_score) = self.get_acquisition_score(
            candidate_tasks,
            real_batch_size=int(shape[0] * shape[1]),
            diversified=True,
        )

        tasks = candidate_tasks[best_batch_id]
        tasks = tasks.view(shape[0], shape[1], shape[2]).numpy()
        return tasks, diversified_score, combine_local_diverse_score, combine_local_acquisition_score

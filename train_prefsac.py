import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import shutil
from offpolicy.video import VideoRecorder
from offpolicy.logger import Logger
from offpolicy.replay_buffer import ReplayBuffer
from offpolicy.agent.sac import SACAgent
import offpolicy.utils as utils
from crowd_nav.configs.config import Config
import crowd_sim
import os
from collections import deque
from preference.reward_model import RewardModel

class Workspace(object):
    def __init__(self):

        self.config = Config()
        utils.set_seed_everywhere(self.config.env.seed)

        env_name = self.config.env.env_name
        task = self.config.env.task
        policy_name = self.config.robot.policy + '_sac'
        self.output_dir = os.path.join(self.config.training.output_dir, task, policy_name)
        # save policy to output_dir
        if os.path.exists(self.output_dir) and self.config.training.overwrite:  # if I want to overwrite the directory
            shutil.rmtree(self.output_dir)  # delete an entire directory tree

        #print(self.output_dir)

        self.output_dir = "/home/peter/Nav/SAN-NaviSTAR-master/output"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        shutil.copytree('crowd_nav/configs', os.path.join(self.output_dir, 'configs'))

        self.model_dir = os.path.join(self.output_dir, 'checkpoints')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.work_dir = self.output_dir
        print(f'workspace: {self.work_dir}')

        self.logger = Logger(self.work_dir,
                             save_tb=True,
                             log_frequency=10000,
                             agent='sac')


        self.device = torch.device("cuda" if self.config.training.cuda and torch.cuda.is_available() else "cpu")
        self.env = utils.make_env(self.config)
        self.eval_env = utils.make_eval_env(self.config)

        obs_shape = self.env.observation_space.spaces
        action_shape = self.env.action_space.shape
        self.agent = SACAgent(self.config, obs_shape, action_shape, self.device)

        if self.config.training.resume:  # retrieve the models if resume = True
            load_path = self.config.training.load_path
            self.agent.actor.load_state_dict(torch.load(load_path))
            print("Loaded the following checkpoint:", load_path)
            load_path = '/home/peter/Nav/SAN-NaviSTAR-master/data/navigation/star_sac/checkpoints/sac_critic594640.pt'
            self.agent.critic.load_state_dict(torch.load(load_path))
            load_path = '/home/peter/Nav/SAN-NaviSTAR-master/data/navigation/star_sac/checkpoints/sac_critic_target594640.pt'
            self.agent.critic_target.load_state_dict(torch.load(load_path))

        self.replay_buffer = ReplayBuffer(obs_shape,
                                          action_shape,
                                          int(self.config.sac.num_train_steps),
                                          self.device)
        pref_obs_shape = [self.config.trans.hidden_size]
        self.preference_replay_buffer = ReplayBuffer(obs_shape,
                                          action_shape,
                                          int(self.config.sac.num_train_steps),
                                          self.device,
                                          obs_pref_shape=pref_obs_shape)

        self.reward_model = RewardModel(
            ds=self.config.trans.hidden_size,
            da=self.env.action_space.shape[0],
            ensemble_size=self.config.preference.ensemble_size,
            size_segment=self.config.preference.segment,
            activation=self.config.preference.activation,
            lr=self.config.preference.reward_lr,
            mb_size=self.config.preference.reward_batch,
            large_batch=self.config.preference.large_batch,
            label_margin=self.config.preference.label_margin,
            teacher_beta=self.config.preference.teacher_beta,
            teacher_gamma=self.config.preference.teacher_gamma,
            teacher_eps_mistake=self.config.preference.teacher_eps_mistake,
            teacher_eps_skip=self.config.preference.teacher_eps_skip,
            teacher_eps_equal=self.config.preference.teacher_eps_equal)

        self.video_recorder = VideoRecorder(
            self.work_dir if self.config.sac.save_video else None)
        self.step = 0
        self.interaction = 0

        self.max_success_rate = 0.

        self.step = 594640


    def evaluate(self):
        success = 0
        collision = 0
        timeout = 0
        average_episode_reward = 0
        for episode in range(self.config.sac.num_eval_episodes):
            self.eval_env.case_counter['test'] = 0
            obs = self.eval_env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.eval_env.step(action)
                done = done[0]
                episode_reward += reward[0]

            average_episode_reward += episode_reward
            status = str(info['info'])
            if status == 'Reaching goal':
                success += 1
            elif status == 'Collision':
                collision += 1
            elif status == 'Timeout':
                timeout += 1
        average_episode_reward /= self.config.sac.num_eval_episodes
        success_rate = success / self.config.sac.num_eval_episodes
        collision_rate = collision / self.config.sac.num_eval_episodes
        timeout_rate = timeout / self.config.sac.num_eval_episodes

        if success_rate > self.max_success_rate:
            self.max_success_rate = success_rate
            torch.save(self.agent.actor.state_dict(), os.path.join(self.model_dir, './best_sac_actor.pt'))


        print('eval', average_episode_reward)
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/success_rate', success_rate,
                        self.step)
        self.logger.log('eval/collision_rate', collision_rate,
                        self.step)
        self.logger.log('eval/timeout_rate', timeout_rate,
                        self.step)
        self.logger.dump(self.step)

    def learn_reward(self, first_flag=0):

        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.config.preference.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.config.preference.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.config.preference.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.config.preference.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.config.preference.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.config.preference.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError

        train_acc = 0
        if labeled_queries > 0:
            # update reward
            for epoch in range(self.config.preference.reward_update):
                if self.config.preference.label_margin > 0 or self.config.preference.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)

                if total_acc > 0.97:
                    break

        print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode, episode_step, episode_reward, done = 0, 0, 0, True
        start_time = time.time()
        episode_rewards = deque(maxlen=100)
        reward_list = []
        while self.step < self.config.sac.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.config.sac.num_seed_steps))

                if self.interaction > self.config.sac.save_interval:
                    self.interaction = 0
                    filename = 'sac_actor' + str(self.step) + '.pt'
                    torch.save(self.agent.actor.state_dict(), os.path.join(self.model_dir, filename))

                    filename = 'sac_critic' + str(self.step) + '.pt'
                    torch.save(self.agent.critic.state_dict(), os.path.join(self.model_dir, filename))

                    filename = 'sac_critic_target' + str(self.step) + '.pt'
                    torch.save(self.agent.critic_target.state_dict(), os.path.join(self.model_dir, filename))

                    np.save(os.path.join(self.output_dir, 'reward.npy'), reward_list)
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)
                if self.step >= self.config.sac.num_seed_steps:
                    episode_rewards.append(episode_reward)
                    reward_list.append(np.mean(episode_rewards))
                    print('%d/%d, %d, %f' % (self.step, self.config.sac.num_train_steps, episode_step, np.mean(episode_rewards)))

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.config.sac.num_seed_steps:
                action = self.env.action_space.sample()
                action = utils.clip_action(action, clip_norm=True, max_norm=self.config.robot.v_pref)
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                    action = utils.clip_action(action, clip_norm=True, max_norm=self.config.robot.v_pref)

            # run training update
            if self.step >= self.config.sac.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)


            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done[0])
            done_no_max = done
            episode_reward += reward[0]

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            self.interaction += 1


        episode_step, episode_reward, done = 0, 0, True
        self.interaction = 0
        self.preference_interaction = 0
        print('start preference')

        while self.step < self.config.sac.num_train_steps + self.config.preference.training_steps:
            if done:
                if self.step > self.config.sac.num_train_steps:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.config.sac.num_train_steps + self.config.preference.init_steps))

                if self.interaction > self.config.sac.save_interval:
                    self.interaction = 0
                    filename = 'sac_actor' + str(self.step) + '.pt'
                    torch.save(self.agent.actor.state_dict(), os.path.join(self.model_dir, filename))
                    np.save(os.path.join(self.output_dir, 'reward.npy'), reward_list)
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)
                if self.step >= self.config.sac.num_train_steps + self.config.preference.init_steps:
                    episode_rewards.append(episode_reward)
                    reward_list.append(np.mean(episode_rewards))
                    print('%d/%d, %d, %f' % (
                    self.step, self.config.sac.num_train_steps, episode_step, np.mean(episode_rewards)))

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            if self.preference_interaction >= self.config.preference.num_interaction:
                # # update schedule
                # if self.cfg.reward_schedule == 1:
                #     frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                #     if frac == 0:
                #         frac = 0.01
                # elif self.cfg.reward_schedule == 2:
                #     frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                # else:
                #     frac = 1
                # self.reward_model.change_batch(frac)

                # # update margin --> not necessary / will be updated soon
                # new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                # self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                # self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)

                # # corner case: new total feed > max feed
                # if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                #     self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)

                self.learn_reward()
                self.preference_replay_buffer.relabel_with_predictor(self.reward_model)
                self.preference_interaction = 0


            with utils.eval_mode(self.agent):
                action = self.agent.act(obs, sample=True)
                action = utils.clip_action(action, clip_norm=True, max_norm=self.config.robot.v_pref)
                obs_for_pref = self.agent.actor.obs_features

                # run training update
            if self.step >= self.config.sac.num_train_steps + self.config.preference.init_steps:
                self.agent.update(self.preference_replay_buffer, self.logger, self.step)

            next_obs, reward, done, info = self.env.step(action)
            reward_hat = [self.reward_model.r_hat(np.concatenate([obs_for_pref, action], axis=-1))]

            done = float(done[0])
            done_no_max = done
            episode_reward += reward[0]

            self.reward_model.add_data(obs_for_pref, action, reward, done)
            self.preference_replay_buffer.add(obs, action, reward_hat, next_obs, done,
                                   done_no_max, obs_pref=obs_for_pref)

            obs = next_obs
            episode_step += 1
            self.step += 1
            self.interaction += 1
            self.preference_interaction += 1


def main():
    workspace = Workspace()
    workspace.run()


if __name__ == '__main__':
    main()

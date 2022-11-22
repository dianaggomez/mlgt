import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import time
import os

# plt.style.use('dark_background')


class MAPPOTrainer:
    """
    A class for the implementation and utilization of the training process
    steps for the Multi-Agent Proximal Policy Optimization algorithm.

    Attributes:
        env: A UnityEnvironment used for Agent evaluation and training.
        agents: Agent objects being trained in env.
        score_window_size: Integer window size used in order to gather
            mean score to evaluate environment solution.
        max_epsiode_length: An integer for maximum number of timesteps per
            episode.
        update_frequency: An integer designating the step frequency of
            updating target network parameters.
        save_dir: Path designating directory to save resulting files.
    """

    def __init__(self, env, agents, score_window_size, max_episode_length,
                 update_frequency, save_dir):
        """Initializes MAPPOTrainer attributes."""

        # Initialize relevant variables for training.
        self.env = env
        # self.brain_name = env.brain_names[0]
        self.agents = agents
        self.score_window_size = score_window_size
        self.max_episode_length = 60 #max_episode_length, stop after 60 iterations
        self.update_frequency = update_frequency
        self.save_dir = save_dir
        self.score_history = []
        self.episode_length_history = []
        self.timestep = 0
        self.i_episode = 0

    def step_env(self, actions):
        """
        Realizes actions in environment and returns relevant attributes.

        Parameters:
            actions: Actions array to be realized in the environment.

        Returns:
            states: Array with next state information.
            rewards: Array with rewards information.
            dones: Array with boolean values with 'true' designating the
                episode has finished.
            env_info: BrainInfo object with current environment data.
        """

        # From environment information, extract states and rewards.
        obs, share_obs, rewards, done, infos = self.env.step(actions)
        states = share_obs # double check this

        # Evaluate if episode has finished.
        dones = done

        return states, rewards, dones, infos #, env_info

    def run_episode(self):
        """
        Runs a single episode in the training process for max_episode_length
        timesteps.

        Returns:
            scores: List of rewards gained at each timestep.
        """

        # Initialize list to hold reward values at each timestep.
        scores = []

        # Restart the environment and gather original states.
        obs, share_obs = self.env.reset()
        states = share_obs
        # print(len(states))
        # Act and evaluate results and networks for each timestep.
        for t in range(self.max_episode_length):
            self.timestep += 1

            # Sample actions for each agent while keeping track of states,
            # actions and log probabilities.
            processed_states, actions, log_probs = [], [], []
            for agent, state in zip(self.agents, states):
                # print(state)
                processed_state = torch.from_numpy(state).float()
                action, log_prob = agent.get_actions(processed_state)
                processed_states.append(processed_state)
                log_probs.append(log_prob)
                actions.append(action)

            # Convert action tensors to integer values.
            # raw_actions = np.array(
            #     [torch.clamp(a, -1, 1).numpy() for a in actions]
            # )
            # print(actions)
            # print(type(actions))
            raw_actions = np.array(
                [torch.clamp(a, -1, 1).numpy()  for a in actions]
            )
            # print(raw_actions)
            # Realize sampled actions in environment and evaluate new state.
            # states, rewards, dones, _ = self.step_env(raw_actions)

            # TODO: Figure out if the clipping needs to be completed
            # convert the sampled actions to the bound of each action
            
            # gather high and low for each
            low = self.env.action_space[0].low
            high = self.env.action_space[0].high
            # x = [ (x - min(x)) / (max(x) - min(x)) - 0.5] * 2
            raw_actions = np.array(
                [(a/2 + 0.5)*(high - low) + low for a in raw_actions])
            # actions = np.array(
            #     [(a - low) / (high - low) - 0.5 for a in actions]
            # ) * 2

            # print(actions)
            obs, share_obs, rewards, dones, infos = self.env.step(raw_actions)
            # obs, share_obs, rewards, dones, infos = self.env.step(raw_actions)
            states = share_obs # double check this


            # Add experience to the memories for each agent.
            for agent, state, action, log_prob, reward, done in \
                    zip(self.agents, processed_states, actions, log_probs,
                        rewards, dones):
                agent.add_memory(state, action, log_prob, reward, done)

            # Initiate learning for agent if update frequency is observed.
            if self.timestep % self.update_frequency == 0:

            # TODO: Only update the agent that we are training
                # for agent in self.agents:
                #     agent.update()

                self.agents[0].update()

            # Append reward gained for each new action.
            scores.append(rewards)

            # End episode if desired score is achieved.
            # if np.any(dones):
            #     break
            if np.all(dones):
                break

        return scores

    def step(self):
        """
        Initiates run of an episode and logs the resulting total rewards and
        episode lengths.
        """

        # Run a single episode in environment.
        self.i_episode += 1
        scores = self.run_episode()
        # print(
        #     "Scores ", scores
        # )
        # Sum the episode rewards for each agent to get the total rewards.
        score_by_agent = np.sum(np.array(scores), axis=0)
        # print(
        #     "Sum Score ", score_by_agent
        # )
        # Store total rewards and episode lengths.
        self.score_history.append(score_by_agent)
        # print(
        #     "History ", self.score_history
        # )
        self.episode_length_history.append(len(scores))

    def save(self):
        """
        Saves actor_critic for both agents once successful score is achieved.
        """

        # Save actor_critic for each agent in specified save location.
        for agent_ix in range(len(self.agents)):
            agent = self.agents[agent_ix]
            filename = f'agent_{agent_ix}_episode_{self.i_episode}_1_constant.pth'
            state_dict = agent.actor_critic.state_dict()
            torch.save(state_dict, os.path.join(self.save_dir, filename))

    def print_status(self):
        """Prints reward info and episode length stats at current episode."""
   
        # Calculate necessary statistics.
        mean_reward = np.mean(
            self.score_history[-self.score_window_size:],
            axis=0
        )
        # print(len(mean_reward))
        # agent_info = ['Mean Reward Agent 0: ', mean_reward[0], 'Mean Reward Agent 1: ', mean_reward[1]]
        agent_info = ''.join(f'Mean Reward Agent_{i}: {mean_reward[i]:.2f}, '
                             for i in range(len(self.agents)))
        max_mean = np.max(self.score_history[-self.score_window_size:],
                          axis=1).mean()
        mean_eps_len = np.mean(
            self.episode_length_history[-self.score_window_size:]
        ).item()

        # Print current status to terminal.
        print(
            f'\033[1mEpisode {self.i_episode} - '
            f'Mean Max Reward: {max_mean:.2f}\033[0m'
            f'\n\t{agent_info}\n\t'
            f'Mean Total Reward: {mean_reward.sum():.2f}, '
            f'Mean Episode Length {mean_eps_len:.1f}'
        )

    def plot(self):
        """
        Plots moving averages of maximum reward and rewards for each agent.
        """

        # Initialize DataFrame to be used for plot.
        columns = [f'Agent {i}' for i in range(len(self.agents))]
        df = pd.DataFrame(self.score_history, columns=columns)
        df['Max'] = df.max(axis=1)

        # Plot rewards per agent and cumulative moving averages.
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_title(
            f'Learning Curve: Multi-Agent PPO',
            fontsize=28
        )
        ax.set_xlabel('Episode', fontsize=21)
        ax.set_ylabel('Reward', fontsize=21)
        df.rolling(self.score_window_size).mean()\
            .plot(ax=ax)
            # .plot(ax=ax, colormap='terrain')
        # ax.grid(color='w', linewidth=0.2)
        ax.grid(linewidth=0.2)
        ax.legend(fontsize=13)
        plt.tight_layout()

        # Save resulting plot.
        filename = f'reward_{self.i_episode}'
        fig.savefig(os.path.join(self.save_dir, filename))
        plt.show()

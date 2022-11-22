from mappo.mappo_trainer import MAPPOTrainer
# from unityagents import UnityEnvironment
from OneShot_Env import OneShotEnv
from config import get_config
from mappo.ppo_model import PolicyNormal
from mappo.ppo_model import CriticNet
from mappo.ppo_model import ActorNet
from mappo.ppo_agent import PPOAgent
import numpy as np
import torch
import sys
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_args(args, parser):
    parser.add_argument('--oneshot_name', type=str,
                        default='OneShot-Very-Small', help="Which env to run on")
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def load_env(args):
    """
    Initializes the UnityEnviornment and corresponding environment variables
    based on the running operating system.

    Arguments:
        env_loc: A string designating unity environment directory.

    Returns:
        env: A UnityEnvironment used for Agent evaluation and training.
        num_agents: Integer number of Agents to be trained in environment.
        state_size: Integer number of possible states.
        action_size: Integer number of possible actions.
    """
    env = OneShotEnv(args, args.seed)
    # Extract state dimensionality from env.
    state_size = env.observation_space[0].shape[0]

    # Extract action dimensionality and number of agents from env.
    action_size = env.action_space[0].shape[0]
    num_agents = env.players

    # Display relevant environment information.
    print('\nNumber of Agents: {}, State Size: {}, Action Size: {}\n'.format(
        num_agents, state_size, action_size))

    return env, num_agents, state_size, action_size


def create_agent(state_size, action_size, actor_fc1_units=512,
                 actor_fc2_units=256, actor_lr=1e-4, critic_fc1_units=512,
                 critic_fc2_units=256, critic_lr=1e-4, gamma=0.99,
                 num_updates=10, max_eps_length=500, eps_clip=0.3,
                 critic_loss=0.5, entropy_bonus=0.01, batch_size=256):
    """
    This function creates an agent with specified parameters for training.

    Arguments:
        state_size: Integer number of possible states.
        action_size: Integer number of possible actions.
        actor_fc1_units: An integer number of units used in the first FC
            layer for the Actor object.
        actor_fc2_units: An integer number of units used in the second FC
            layer for the Actor object.
        actor_lr: A float designating the learning rate of the Actor's
            optimizer.
        critic_fc1_units: An integer number of units used in the first FC
            layer for the Critic object.
        critic_fc2_units: An integer number of units used in the second FC
            layer for the Critic object.
        critic_lr: A float designating the learning rate of the Critic's
            optimizer.
        gamma: A float designating the discount factor.
        num_updates: Integer number of updates desired for every
            update_frequency steps.
        max_eps_length: An integer for maximum number of timesteps per
            episode.
        eps_clip: Float designating range for clipping surrogate objective.
        critic_loss: Float designating initial Critic loss.
        entropy_bonus: Float increasing Actor's tendency for exploration.
        batch_size: An integer for minibatch size.

    Returns:
        agent: An Agent object used for training.
    """

    # Create Actor/Critic networks based on designated parameters.
    actor_net = ActorNet(state_size, action_size, actor_fc1_units,
                         actor_fc2_units).to(device)
    critic_net = CriticNet(state_size, critic_fc1_units, critic_fc2_units)\
        .to(device)

    # Create copy of Actor/Critic networks for action prediction.
    actor_net_old = ActorNet(state_size, action_size, actor_fc1_units,
                             actor_fc2_units).to(device)
    critic_net_old = CriticNet(state_size, critic_fc1_units, critic_fc2_units)\
        .to(device)
    actor_net_old.load_state_dict(actor_net.state_dict())
    critic_net_old.load_state_dict(critic_net.state_dict())

    # Create PolicyNormal objects containing both sets of Actor/Critic nets.
    actor_critic = PolicyNormal(actor_net, critic_net)
    actor_critic_old = PolicyNormal(actor_net_old, critic_net_old)

    # Initialize optimizers for Actor and Critic networks.
    actor_optimizer = torch.optim.Adam(
        actor_net.parameters(),
        lr=actor_lr
    )
    critic_optimizer = torch.optim.Adam(
        critic_net.parameters(),
        lr=critic_lr
    )

    # Create and return PPOAgent with relevant parameters.
    agent = PPOAgent(
        device=device,
        actor_critic=actor_critic,
        actor_critic_old=actor_critic_old,
        gamma=gamma,
        num_updates=num_updates,
        eps_clip=eps_clip,
        critic_loss=critic_loss,
        entropy_bonus=entropy_bonus,
        batch_size=batch_size,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer
    )

    return agent


def create_trainer(env, agents, save_dir, update_frequency=5000,
                   max_eps_length=500, score_window_size=100):
    """
    Initializes trainer to train agents in specified environment.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        agents: Agent objects used for training.
        save_dir: Path designating directory to save resulting files.
        update_frequency: An integer designating the step frequency of
            updating target network parameters.
        max_eps_length: An integer for maximum number of timesteps per
            episode.
        score_window_size: Integer window size used in order to gather
            max mean score to evaluate environment solution.

    Returns:
        trainer: A MAPPOTrainer object used to train agents in environment.
    """

    # Initialize MAPPOTrainer object with relevant arguments.
    trainer = MAPPOTrainer(
        env=env,
        agents=agents,
        score_window_size=score_window_size,
        max_episode_length=max_eps_length,
        update_frequency=update_frequency,
        save_dir=save_dir
    )

    return trainer


def train_agents(env, trainer, n_episodes=8000, target_score=0.5,
                 score_window_size=100):
    """
    This function carries out the training process with specified trainer.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        trainer: A MAPPOTrainer object used to train agents in environment.
        n_episodes: An integer for maximum number of training episodes.
        target_score: An float max mean target score to be achieved over
            the last score_window_size episodes.
        score_window_size: The integer number of past episode scores
            utilized in order to calculate the current mean scores.
    """
    n_episodes=10000

    # Train the agent for n_episodes.
    for i_episode in range(1, n_episodes + 1):

        # Step through the training process.
        trainer.step()

        # Print status of training every 100 episodes.
        if i_episode % 100 == 0:
            scores = np.max(trainer.score_history, axis=1).tolist()
            trainer.print_status()

        # If target achieved, print and plot reward statistics.
        mean_reward = np.max(
            trainer.score_history[-score_window_size:], axis=1
        ).mean()
        
        # === Removed DGG =====
        # if mean_reward >= target_score: # need to address this!!!
        #     print('Environment is solved.')
        #     env.close()
        #     trainer.print_status()
        #     trainer.plot()
        #     trainer.save()
        #     break
   # If target achieved, print and plot reward statistics.
    # mean_reward = np.max(
    #     trainer.score_history[-score_window_size:], axis=1
    # ).mean()
    env.previous_action # print the last action
    env.close()
    trainer.print_status()
    trainer.plot()
    trainer.save()

if __name__ == '__main__':
    parser = get_config()
    args = sys.argv[1:]
    all_args = parse_args(args, parser)

    #place sseds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    # Initialize environment, extract state/action dimensions and num agents.
    env, num_agents, state_size, action_size = load_env(all_args)

    # Initialize agents for training.
    agents = [create_agent(state_size, action_size) for _ in range(num_agents)]

    # # Give agent 1 the policy learned
    # for i in range(2):
    #     agents[i].actor_critic.load_state_dict(torch.load("/Users/diana/Desktop/mappo-competitive-reinforcement-main/project_training_files/agent_{num}_episode_10000.pth".format(num = i)))
    #     agents[i].actor_critic.eval()
    # # agents[1].actor_critic.load_state_dict(torch.load("/Users/diana/Desktop/mappo-competitive-reinforcement-main/project_training_files/agent_1_episode_10000.pth"))
    # # agents[1].actor_critic.eval()
    # # Create MAPPOTrainer object to train agents.
    # save_dir = os.path.join(os.getcwd(), r'project_training_files')
    # trainer = create_trainer(env, agents, save_dir)

    # # Train agent in specified environment.
    # train_agents(env, trainer)

    for i in range(2):
        agents[i].actor_critic.load_state_dict(torch.load("/Users/diana/Desktop/mappo-competitive-reinforcement-main/project_training_files/agent_{num}_episode_10000_1_constant.pth".format(num = i)))
        agents[i].actor_critic.eval()

    #     # Print model's state_dict
    #     print("Model's state_dict:")
    #     for param_tensor in agents[i].actor_critic.state_dict():
    #         print(param_tensor, "\t", agents[i].actor_critic.state_dict()[param_tensor].size())

    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    # Given a state (the dummy state) gather the action for each agent

    obs, share_obs = env.reset()
    processed_state = torch.from_numpy(obs).float()

    # print(env.action_space[0].high)
    # print(env.action_space[0].low)
    for j in range(2):
        print(j)
        actions = []
        # print("Normal distribution : ", agents[i].actor_critic.actor(processed_state))
        for i in range(2):
            actions.append(agents[i].get_actions(processed_state)[0])

        raw_actions = np.array(
            [torch.clamp(a, -1, 1).numpy()  for a in actions]
        )
        # gather high and low for each
        low = env.action_space[0].low
        high = env.action_space[0].high
        raw_actions = np.array(
            [(a/2 + 0.5)*(high - low) + low for a in raw_actions])
        # print(np.clip(actions[0], env.action_space[0].low, env.action_space[0].high))
        print("Raw Actions: ", raw_actions)
            # print("Action of Agent {num}: ".format(num = i), agents[i].get_actions(processed_state)[0])
            # print("Log Prob of Action for Agent {num}: ".format(num = i), agents[i].get_actions(processed_state)[1])
            


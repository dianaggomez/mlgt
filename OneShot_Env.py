from gym.spaces import Discrete, Box
import numpy as np

# -------------------------------------------------------------------------------
# Environment API
# -------------------------------------------------------------------------------


class Environment(object):
    """Abstract Environment interface.

    All concrete implementations of an environment should derive from this
    interface and implement the method stubs.
    """

    def seed(self, seed):
        raise NotImplementedError("Not implemented in Abstract Base class")

    def reset(self, config):
        """Reset the environment with a new config.

        Signals environment handlers to reset and restart the environment using
        a config dict.

        Args:
          config: dict, specifying the parameters of the environment to be
            generated.

        Returns:
          observation: A dict containing the full observation state.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def step(self, action):
        """Take one step in the game.

        Args:
          action: dict, mapping to an action taken by an agent.

        Returns:
          observation: dict, Containing full observation state.
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.

        Raises:
          AssertionError: When an illegal action is provided.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def close(self):
        """Take one step in the game.

        Raises:
          AssertionError: abnormal close.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")


class OneShotEnv(Environment):
    """RL interface to a Hanabi environment.

    ```python

    environment = rl_env.make()
    config = { 'players': 5 }
    observation = environment.reset(config)
    while not done:
        # Agent takes action
        action =  ...
        # Environment take a step
        observation, reward, done, info = environment.step(action)
    ```
    """
    # These are parameters that are global and static (at least for now)
    # Market parameters
    market_size = 100 # k

    # consumer preference
    psi  = 0.105 #scaling factor
    
    # uncertainty of the consumer willingness to pay:
    mu_beta = -1.47
    sigma_beta = 1.11
    draws_size = 100

    # Vehicle attributes

    omega = 12.110 # initial manufacturing cost ($k)
    gamma = 8.18 # initial operating cost (cents/mi)
    tau = 241.6 # initial emissions (g/mi)
    alpha = 11.554 # operating cost slope

    #threshold for NE
    epslion = 0.0001

    def __init__(self, args, seed):
        """Creates an environment with the given game configuration.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values.
              - players: int, Number of players \in [2,5].
              - hand_size: int, Hand size \in [4,5].
              - max_information_tokens: int, Number of information tokens (>=0).
              - max_life_tokens: int, Number of life tokens (>=1).
              - observation_type: int.
                0: Minimal observation.
                1: First-order common knowledge observation.
              - seed: int, Random seed.
              - random_start_player: bool, Random start player.
        """
        self._seed = seed
        # max:action 48 obs=1380 min:action=20 obs=783 score=25

        if args.oneshot_name == "One-Shot-SimpleCase":  # max:action=28 obs=215 min:action=10 obs=116 score=5
            config = {
                "players": args.num_agents,
                "vehicles": 1,
                # "observation_type": pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value,
                "seed": self._seed
            }
        elif args.oneshot_name == "One-Shot-3Firms":
            config = {
                "players": 3,
                "vehicles": 1,
                # "observation_type": pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value,
                "seed": self._seed
            }
        else:
            raise ValueError("Unknown environment {}".format(args.oneshot_name))

        assert isinstance(config, dict), "Expected config to be of type dict."

        self.beta = np.random.normal(self.mu_beta , self.sigma_beta, self.draws_size)
        self.players = config['players']
        self.action_space = []
        self.observation_space = []

        self.share_observation_space = []

        for i in range(self.players):
            self.action_space.append(Box(np.array([0, 0]), np.array([0.50, 0.75]))) #  price (p), efficiency reduction(x)), # Bounds: p_j = [0, 50],  x_j = [0, 0.75] 
            self.observation_space.append([0])
            self.share_observation_space.append([0])
            
        # turn them in to array
        self.observation_space = np.array(self.observation_space)
        self.share_observation_space  = np.array(self.share_observation_space)
        # self.previous_action = np.zeros((len(np.array(self.action_space).shape), 2)) # can adjust the 2 for the more options on vehicles
        self.previous_action = np.zeros((len(self.action_space), 2))

        # I believe we dont even need to create an observation space

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def reset(self, choose=True):
        """Resets the environment for a new game.
            Will initialize the parameters
        Returns:
          observation: [] will be empty because we are not using any state 
        """
        # initialize parameters
        self.beta = np.random.normal(self.mu_beta , self.sigma_beta, self.draws_size)

        obs = [0]
        share_obs = []
        for i in range(self.players):
            share_obs.append([0])

        obs = np.array(obs)
        share_obs = np.array(share_obs)

        # available_actions = np.ones((2,))

        self.previous_action = np.zeros((len(self.action_space), 2)) # can adjust the 2 for the more options on vehicles
        
        return obs, share_obs #, available_actions

    def close(self):
        pass


    def step(self, action):
        """Take one step in the game.

        Args:
          action: is a array a = [[p_0,x_0], ...., [p_n,x_n]]
              p = price
              x = efficiency reduction

        Returns:
          observation: [] empty because we have not state
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.

        Raises:
          AssertionError: When an illegal action is provided.
        """
        #### print actions that are intialy being sent through
        # print("actions sent through: ", action)
        # Calculate each firms reward given the action
        # processed_action = []
        # for a in action:
        #     processed_action.append(a.numpy())

        # processed_action = np.array(processed_action)
        # print("Actions: ", action)
        # print(type(processed_action))
        p = action[:,0]*100 # price array containing all agents decisions/actions, nx1 array 
        x = action[:,1] # efficiency increase array containing all agents, nx1 array

        # ==  Dependent == 
        manufacturing_cost = self.alpha * x + self.omega
        oc = self.gamma*(1-x) # operating cost
        ghg = self.tau*(1-x) # green house gases


        # == Exogeneous == 
        # Market - repelem (all param) to match num draws, n x size_draws
        beta_market = np.tile(self.beta,[len(action),1]) # beta,len(beta),len(p)
        x_market = np.transpose(np.tile(x,[len(self.beta),1]))
        p_market = np.transpose(np.tile(p,[len(self.beta),1]))

        # calculate elementwise for (firm, consumer)
        utility = self.psi * (beta_market * self.gamma * (1-x_market) - p_market) # proxy variable to model choice probability, consumer utility
        demand_share = np.mean(np.exp(utility)/np.sum(np.exp(utility),axis = 0),axis = 1) # percent of the market, array size of draws
        demand_volume = demand_share * self.market_size # demand share times size of the market

        # Reward function which is profit
        rewards = (p - manufacturing_cost) * demand_volume
      
        # print("Shape of Previous action: ", np.array(self.previous_action).shape)
        # print("Shape of current action: ", np.array(action).shape)
        difference_in_actions = np.linalg.norm(np.array(self.previous_action) - np.array(action), axis =1)
        # print(difference_in_actions)
        # if (difference_in_actions < self.epslion).any():
        #   done = True
        # else:
        #   done = False

        done = difference_in_actions < self.epslion
        # done = True
        # dprint("done: ", done)

        # Reward is score differential. May be large and negative at game end.
        infos = {'Difference between current and previous action': difference_in_actions}

        obs = np.array([0])

        share_obs = []
        for i in range(self.players):
            share_obs.append([0])
        share_obs = np.array(share_obs)

        # available_actions = np.ones((2,))
      
        self.previous_action = action

        # print("Reward: ", rewards)
        return obs, share_obs, rewards, done, infos #, available_actions # are actions availble for the 


class Agent(object):
    """Agent interface.

    All concrete implementations of an Agent should derive from this interface
    and implement the method stubs.


    ```python

    class MyAgent(Agent):
      ...

    agents = [MyAgent(config) for _ in range(players)]
    while not done:
      ...
      for agent_id, agent in enumerate(agents):
        action = agent.act(observation)
        if obs.current_player == agent_id:
          assert action is not None
        else
          assert action is None
      ...
    ```
    """

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values.
              - colors: int, Number of colors \in [2,5].
              - ranks: int, Number of ranks \in [2,5].
              - players: int, Number of players \in [2,5].
              - hand_size: int, Hand size \in [4,5].
              - max_information_tokens: int, Number of information tokens (>=0)
              - max_life_tokens: int, Number of life tokens (>=0)
              - seed: int, Random seed.
              - random_start_player: bool, Random start player.
          *args: Optional arguments
          **kwargs: Optional keyword arguments.

        Raises:
          AgentError: Custom exceptions.
        """
        raise NotImplementedError("Not implemeneted in abstract base class.")

    def reset(self, config):
        r"""Reset the agent with a new config.

        Signals agent to reset and restart using a config dict.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values.
              - colors: int, Number of colors \in [2,5].
              - ranks: int, Number of ranks \in [2,5].
              - players: int, Number of players \in [2,5].
              - hand_size: int, Hand size \in [4,5].
              - max_information_tokens: int, Number of information tokens (>=0)
              - max_life_tokens: int, Number of life tokens (>=0)
              - seed: int, Random seed.
              - random_start_player: bool, Random start player.
        """
        raise NotImplementedError("Not implemeneted in abstract base class.")

    def act(self, observation):
        """Act based on an observation.

        Args:
          observation: dict, containing observation from the view of this agent.
            An example:
            {'current_player': 0,
             'current_player_offset': 1,
             'deck_size': 40,
             'discard_pile': [],
             'fireworks': {'B': 0,
                       'G': 0,
                       'R': 0,
                       'W': 0,
                       'Y': 0},
             'information_tokens': 8,
             'legal_moves': [],
             'life_tokens': 3,
             'observed_hands': [[{'color': None, 'rank': -1},
                             {'color': None, 'rank': -1},
                             {'color': None, 'rank': -1},
                             {'color': None, 'rank': -1},
                             {'color': None, 'rank': -1}],
                            [{'color': 'W', 'rank': 2},
                             {'color': 'Y', 'rank': 4},
                             {'color': 'Y', 'rank': 2},
                             {'color': 'G', 'rank': 0},
                             {'color': 'W', 'rank': 1}]],
             'num_players': 2}]}

        Returns:
          action: dict, mapping to a legal action taken by this agent. The following
            actions are supported:
              - { 'action_type': 'PLAY', 'card_index': int }
              - { 'action_type': 'DISCARD', 'card_index': int }
              - {
                  'action_type': 'REVEAL_COLOR',
                  'color': str,
                  'target_offset': int >=0
                }
              - {
                  'action_type': 'REVEAL_RANK',
                  'rank': str,
                  'target_offset': int >=0
                }
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

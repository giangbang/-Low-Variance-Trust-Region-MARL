import magent
from magent.builtin.tf_model import DeepQNetwork

class MAgent:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.mapsize = self.args["mapsize"]
        self.n_agents = self.args["n_agents"]
        self.n_preys = self.args["n_preys"]
        self.env  = magent.GridWorld("pursuit", map_size=map_size)
        self.env.set_render_dir("build/render")
        self.episode_limit = self.args["episode_limit"]
        self.steps = 0

        # get group handles
        self.predator, self.prey = self.env.get_handles()
        if self.n_agents == "auto":
            self.n_agents = self.map_size * self.map_size * 0.02
        if self.n_preys == "auto":
            self.n_preys = self.map_size * self.map_size * 0.02

        # init env and agents
        self.env.reset()
        self.env.add_walls(method="random", n=self.map_size * self.map_size * 0.01)
        self.env.add_agents(predator, method="random", n=self.n_agents)
        self.env.add_agents(prey, method="random", n=self.n_preys)

        # init two models
        self.model1 = DeepQNetwork(env, predator, "predator")
        self.model2 = DeepQNetwork(env, prey, "prey")

        # load trained model
        self.model1.load("data/pursuit_model")
        self.model2.load("data/pursuit_model")


    def predator_act(self):
        obs_2 = self.env.get_observation(prey)
        ids_2 = self.env.get_agent_id(prey)
        acts_2 = self.model2.infer_action(obs_2, ids_2)
        self.env.set_action(self.prey, acts_2)

    def prey_act(self):
        obs_1 = self.env.get_observation(predator)
        ids_1 = self.env.get_agent_id(predator)
        acts_1 = self.model1.infer_action(obs_1, ids_1)
        self.env.set_action(self.predator, acts_1)

    def step(self, actions):
        self.steps += 1
        self.env.set_action(self.predator, actions)
        self.prey_act()

        done = env.step()
        info = {}

        rewards = sum(env.get_reward(predator))
        env.clear_dead()

        if done:
            if self.steps < self.episode_limit:
                # the next state will be masked out
                info["bad_transition"] = False
            else:
                # the next state will not be masked out
                info["bad_transition"] = True

        return (
            self.env.get_observation(predator),
            self.get_state(),
            [rewards] * self.n_agents,
            [done] * self.n_agents,
            [info] * self.n_agents,
            self.get_avail_actions()
        )

    def get_state(self):
        pass

    def get_avail_actions(self):
        return None
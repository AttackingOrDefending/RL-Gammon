"""Run the trainer."""
import torch as th

from rlgammon.agents.td_agent import TDAgent
from rlgammon.trainer.step_trainer import StepTrainer

if __name__ == "__main__":
    agent = TDAgent(layer_list=[th.nn.Linear(198, 128),
                                th.nn.Linear(128, 128),
                                th.nn.Linear(128, 6),
                                ],
                    activation_list=[th.nn.ReLU,
                                     th.nn.ReLU,
                                     th.nn.Sigmoid,
                                     ],
                    )
    trainer = StepTrainer()
    trainer.load_parameters("parameters.json")
    trainer.train(agent)

    """
    def __init__(self):
        super(DQN, self).__init__()

        self.eligibility_traces = None
        self.fc1 = nn.Linear(198, 128)
        # self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 6)
        self.gamma = .99
        self.lamda = 0.99
        self.lr = 0.01

    def forward(self, x):
        x = th.from_numpy(np.array(x, dtype=np.float32))
        x = th.relu(self.fc1(x))
        # x = th.relu(self.fc2(x))
        x = th.softmax(self.fc3(x), dim=-1)
        x = x[0] * -3 + x[1] * -2 + x[2] * -1 + x[3] * 1 + x[4] * 2 + x[5] * 3
        return x
    """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .harness import ImitationLearningHarness


def initialize_parameters(m):
    # From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class FiLM(nn.Module):
    """Featurewise independent Linear Modulation.

    Taken from BabyAI (Chevalier-Boisvert et al).
    """

    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=imm_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels,
            out_channels=out_features,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class ImageBOWEmbedding(nn.Module):
    """Embed 'image words' in n_channels channels into embeddings.

    Taken from BabyAI (Chevalier-Boisvert et al)."""

    def __init__(self, max_value, n_channels, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_channels * max_value, embedding_dim)
        self.apply(initialize_parameters)

    def forward(self, inputs):
        offsets = torch.Tensor(
            [0, self.max_value, 2 * self.max_value, 3 * self.max_value]
        ).to(inputs.device)
        offsetted = (inputs + offsets[None, None, None, :]).long()
        each_embedding = self.embedding(offsetted)
        each_embedding_flat = each_embedding.reshape(*each_embedding.shape[:-2], -1)

        return each_embedding_flat.permute(0, 3, 1, 2)


class ACModel(nn.Module):
    """Baseline Actor-critic model takne from BabyAI (Chevalier-Boisvert et al).

    The modifciations are:

     1. No memory, since everything is fully observed
     2. The 'bow_endpool_res' architecture is effectively the
        default, remove all the other-architecture-specific paths.
    """

    def __init__(self, n_actions, image_dim=128, instr_dim=128):
        super().__init__()
        # Decide which components are enabled
        self.image_dim = image_dim
        self.instr_dim = instr_dim

        self.word_embeddings = nn.Embedding(16, self.instr_dim)
        self.image_conv = nn.Sequential(
            *[
                ImageBOWEmbedding(16, 4, 128),
                nn.Conv2d(
                    in_channels=128 * 4,
                    out_channels=128,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ]
        )
        self.film_pool = nn.MaxPool2d(kernel_size=(7, 7), stride=2)

        # Define instruction embedding
        gru_dim = self.instr_dim
        self.instr_rnn = nn.GRU(
            self.instr_dim, gru_dim // 2, batch_first=False, bidirectional=True
        )
        self.final_instr_dim = self.instr_dim

        num_module = 2
        self.controllers = []
        for ni in range(num_module):
            mod = FiLM(
                in_features=self.final_instr_dim,
                out_features=128 if ni < num_module - 1 else self.image_dim,
                in_channels=128,
                imm_channels=128,
            )
            self.controllers.append(mod)
            self.add_module("FiLM_" + str(ni), mod)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, n_actions)
        )

        # Define critic's model
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, mission, images_path, directions_path):
        mission_s = mission.transpose(0, 1)
        mission_words = self.word_embeddings(mission_s)
        mission_enc = (
            self.instr_rnn(
                mission_words,
                torch.zeros(
                    2,
                    mission_words.shape[1],
                    mission_words.shape[-1] // 2,
                    device=mission_words.device,
                ),
            )[1]
            .transpose(0, 1)
            .reshape(mission.shape[0], -1)
        )

        concat_observations = torch.cat(
            [
                images_path,
                directions_path[:, :, None, None, None].expand(
                    directions_path.shape[0],
                    images_path.shape[1],
                    images_path.shape[2],
                    images_path.shape[3],
                    1,
                ),
            ],
            dim=-1,
        )

        concat_observations_flat = concat_observations.reshape(
            -1, *concat_observations.shape[-3:]
        )

        # Need to expand the mission_enc too
        mission_enc_seq = mission_enc[:, None].expand(
            mission_enc.shape[0], concat_observations.shape[1], mission_enc.shape[-1]
        )
        mission_enc_flat = mission_enc_seq.reshape(-1, mission_enc_seq.shape[-1])

        x = self.image_conv(concat_observations_flat)
        for controller in self.controllers:
            out = controller(x, mission_enc_flat)
            out = out + x
            x = out
        x = F.relu(self.film_pool(x))
        embedding = x.reshape(x.shape[0], -1)
        embedding_seq = embedding.reshape(*concat_observations.shape[:2], -1)

        actor_dist = F.log_softmax(self.actor(embedding_seq), dim=-1)
        critic_dist = self.critic(embedding_seq).squeeze(-1)

        return actor_dist, critic_dist


class ACModelImitationLearningHarness(ImitationLearningHarness):
    def __init__(self, lr=10e-4, entropy_bonus=10e-3):
        super().__init__(lr=lr, entropy_bonus=entropy_bonus)
        self.policy_model = ACModel(7)

    def forward(self, x):
        mission, images_path, directions_path = x
        return self.policy_model(mission, images_path, directions_path)

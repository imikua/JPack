import jittor as jt
from jittor import nn

from tapnet.model_backend import cat, softmax, xavier_uniform_
from tapnet.models.attention import AttnModule, SelfAttention
from tapnet.models.encoder import SpaceEncoder, obs_to_tensor
from tapnet.models.greedy import Greedy
from tapnet.models.network import Actor


class NewActor(nn.Module):
    def __init__(
        self,
        checkpoint_list,
        box_dim,
        ems_dim,
        hidden_dim,
        prec_dim,
        encoder_type,
        stable_predict,
        corner_num,
        device,
        container_height=1,
    ) -> None:
        super().__init__()

        self.space_encoder = SpaceEncoder(ems_dim, hidden_dim, corner_num)
        self.space_decoder = AttnModule(
            hidden_dim, heads=8, dropout=0, device=device, max_length=300, merge_flag=False
        )
        self.state_encoder = SpaceEncoder(ems_dim + box_dim, hidden_dim, corner_num=1)
        self.attn = SelfAttention(hidden_dim, heads=8)
        self.attn_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        for p in self.parameters():
            if len(p.shape) > 1:
                xavier_uniform_(p)

        self.actor_list = nn.ModuleList()
        for checkpoint in checkpoint_list or []:
            if checkpoint is None:
                continue

            if "greedy" in checkpoint:
                actor = Greedy(pack_type="last", container_height=container_height, device=device)
            else:
                actor = Actor(
                    box_dim,
                    ems_dim,
                    hidden_dim,
                    prec_dim,
                    encoder_type,
                    stable_predict,
                    corner_num,
                    device,
                )
                print("Loading ", checkpoint)
                actor.load(checkpoint)

            actor.eval()
            self.actor_list.append(actor)

        if len(self.actor_list) == 0:
            raise ValueError("`checkpoint_list` must contain at least one valid checkpoint for `tnex`.")

        self.device = device

    def decode_actions(self, actions, batch_size, ems_num, box_state_num, box_states, ems):
        action_num = ems_num * box_state_num
        corner_ptr = actions % action_num

        ems_dim = ems.shape[2]
        box_dim = box_states.shape[2]

        ems_id = (corner_ptr // box_state_num).reshape((-1, 1, 1)).expand((batch_size, 1, ems_dim)).astype(jt.int32)
        box_state_id = (corner_ptr % box_state_num).reshape((-1, 1, 1)).expand((batch_size, 1, box_dim)).astype(jt.int32)

        new_box_states = jt.gather(box_states, 1, box_state_id)
        new_ems = jt.gather(ems, 1, ems_id)
        return cat([new_box_states, new_ems], dim=2)

    def execute(self, obs, state=None, info={}):
        all_probs = []

        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, _, _ = obs_to_tensor(
            obs, self.device
        )
        batch_size = len(box_states)

        ems_vecs = self.space_encoder(ems)
        for ems_vec in ems_vecs:
            container_vecs = self.space_decoder(ems_vec, ems_mask)

        all_new_states = []
        for actor in self.actor_list:
            probs, state = actor(obs, state=None, info={})
            actions = probs.argmax(dim=1)
            new_states = self.decode_actions(actions, batch_size, ems_num, box_state_num, box_states, ems)
            all_new_states.append(new_states)
            all_probs.append(probs)

        all_new_states = cat(all_new_states, dim=1)
        state_vecs = self.state_encoder(all_new_states)[0]

        vecs = self.attn(container_vecs, container_vecs, state_vecs, None)
        attn_vecs = self.attn_embedding(vecs).reshape(batch_size, -1)
        attn_scores = softmax(attn_vecs, dim=1)

        final_probs = jt.zeros_like(all_probs[0])
        for i, probs in enumerate(all_probs):
            final_probs += attn_scores[:, i : i + 1] * probs

        return final_probs, state

    def forward(self, obs, state=None, info={}):
        return self.execute(obs, state, info)
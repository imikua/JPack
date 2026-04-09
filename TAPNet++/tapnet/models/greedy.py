import jittor as jt
from jittor import nn
from tapnet.models.encoder import obs_to_tensor
from tapnet.model_backend import masked_fill, ones, softmax

class Greedy(nn.Module):
    def __init__(self, pack_type, container_height, device, *kwargs) -> None:
        super(Greedy, self).__init__()
        self.greedy = nn.Linear(1,1)
        self.device = device
        self.pack_type = pack_type
        self.container_height = container_height
    
    def execute(self, obs, state=None, info={} ):
        # comapre max reawrd: size / max_h

        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, _, _ = obs_to_tensor(obs, self.device)

        batch_size = len(box_states)

        # pos | size
        # ems: batch x ems_num x 6
        # box_states: batch x (rot_num * axis_num * box_num) x 3
        
        prec_mask = access_mask.clone() * 1

        if False:
            max_prec = prec_states.sum(dim=3).sum(dim=1) # batch x box_num
            box_num = max_prec.shape[-1]
            state_num = box_state_num / box_num
            
            max_prec[ ~valid_mask[:, :box_num] ] = 0
            max_prec[ ~access_mask[:, :box_num] ] = 0
            # NOTE bug in mask

            max_mask = max_prec == max_prec.max(dim=1)[0].unsqueeze(1)
            # max_mask = max_prec.argmax(dim=1)

            prec_mask = access_mask.clone() * 0
            for s in range(int(state_num)):
                prec_mask[:, s*box_num:(s+1)*box_num][ max_mask ] = 1

        
        # batch x 1 x (rot_num * axis_num * box_num)
        box_z = box_states[:, :, 2].unsqueeze(1)

        # batch x 1 x (rot_num * axis_num * box_num)
        box_size = box_states[:, :, 0] * box_states[:, :, 1] * box_states[:, :, 2]
        box_size = box_size.unsqueeze(1)

        # batch x ems_num x 1
        ems_z = ems[:, :, 2:3]

        ems_h = ems[:, :, 5]
        
        # ems_id = ems[:, :, 6]

        # batch

        # batch x ems_num x box_z
        ems_to_box_height = ems_z + box_z

        if self.pack_type == 'all' or self.pack_type == 'last':
            max_h = jt.max(ems_h, dim=1, keepdims=False)[0].unsqueeze(1).unsqueeze(1)
            max_h = max_h.expand(batch_size, ems_num, box_state_num)

            higher_mask = ems_to_box_height < max_h
            if bool(higher_mask.max().numpy()):
                ems_to_box_height[higher_mask] = max_h[higher_mask]

        # elif self.pack_type == 'last':
        # ems_to_box_height = self.container_height

        # reward
        # sh_rate = box_size / self.container_height
        sh_rate = -ems_to_box_height
        # add mask for valid ems-box pair
        valid_action_mask = (
            ems_to_box_mask.astype(jt.float32)
            * (prec_mask * valid_mask * access_mask).unsqueeze(1).astype(jt.float32)
        )
        height_score = sh_rate + jt.log(valid_action_mask + 1e-12)

        height_score = height_score.reshape(batch_size, -1)
        
        # batch x ...
        probs = softmax(height_score, dim=1)

        # lower is better

        low_ems_to_box_height = (1 - ems_to_box_height) #+ ems_to_box_mask.float().log() + (prec_mask * valid_mask * access_mask).unsqueeze(1).float().log()
        low_ems_to_box_height = low_ems_to_box_height.reshape(batch_size, -1)
        # low_height_probs = F.softmax(low_ems_to_box_height, dim=1)
        low_height_probs = low_ems_to_box_height
        new_probs = probs * low_height_probs
        new_scores = masked_fill(new_probs, valid_action_mask.reshape(batch_size, -1) <= 0, float("-inf"))
        probs = softmax(new_scores, dim=1)
        
        if self.training == False:
            prob_max = probs.max(dim=1, keepdims=True)[0]
            probs[ probs != prob_max ] = 0
            probs[ probs == prob_max ] = 1
            probs /= probs.sum(dim=1, keepdims=True)
        
        return probs, state

    def forward(self, obs, state=None, info={} ):
        return self.execute(obs, state, info)
        

class Critic(nn.Module):
    def __init__(self, device, *kwargs) -> None:
        super(Critic, self).__init__()
        self.greedy = nn.Linear(1,1)
        self.device = device

    def execute(self, obs, act=None, info={} ):
        batch_size = len(obs.box_num)
        output_value = ones((batch_size, 1), dtype=jt.float32)
        return output_value

    def forward(self, obs, act=None, info={} ):
        return self.execute(obs, act, info)
        
    
class Actor(nn.Module):
    def __init__(self, device, pack_type="last", container_height=1, *kwargs) -> None:
        super(Actor, self).__init__()
        self.actor = Greedy(pack_type=pack_type, container_height=container_height, device=device)
    
    def execute(self, obs, state=None, info={} ):
        probs, state = self.actor(obs, state=None, info={})
        return probs, state

    def forward(self, obs, state=None, info={} ):
        return self.execute(obs, state, info)
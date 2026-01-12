import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math
torch.set_default_dtype(torch.float32)




"""
GANs
"""

class GAN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Generator = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            self.Discriminator = MLP(input_dim, 1, hidden_dim, num_layers, 'sigmoid', output_dim)
        elif network == 'att':
            self.Generator = ATT(input_dim, 1, hidden_dim, num_layers, output_act, latent_dim=1, pred_type=pred_type)
            self.Discriminator = ATT(input_dim, 1, hidden_dim, num_layers, 'sigmoid', latent_dim=1, agg=True)
        else:
            NotImplementedError
        self.criterion = nn.BCELoss()

    def forward(self, x, z):
        y_pred = self.Generator(x, z)
        return y_pred

    def loss_g(self, x, y_pred):
        valid = torch.ones([x.shape[0], 1]).to(x.device)
        return self.criterion(self.Discriminator(x, y_pred), valid)

    def loss_d(self, x, y_target, y_pred):
        valid = torch.ones([x.shape[0], 1]).to(x.device)
        fake = torch.zeros([x.shape[0], 1]).to(x.device)
        d_loss = (self.criterion(self.Discriminator(x, y_target), valid) +
                  self.criterion(self.Discriminator(x, y_pred.detach()), fake)) / 2
        return d_loss


"""
DDPM & DDIM
"""
class DM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(DM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.con_dim = input_dim
        self.time_step = time_step
        self.output_dim = output_dim
        beta_max = 0.02
        beta_min = 1e-4
        self.betas = sigmoid_beta_schedule(self.time_step, beta_min, beta_max)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.criterion = nn.L1Loss()
        self.normalize = output_norm

    def predict_noise(self, x, y, t, noise):
        y_t = self.diffusion_forward(y, t, noise)
        noise_pred = self.model(x, y_t, t)
        return noise_pred

    def diffusion_forward(self, y, t, noise):
        if self.normalize:
            y = y * 2 - 1
        t_index = (t * self.time_step).to(dtype=torch.long)
        sqrt_alphas = self.sqrt_alphas_cumprod.to(y.device)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod.to(y.device)
    
        alphas_1 = sqrt_alphas[t_index]
        alphas_2 = sqrt_one_minus_alphas[t_index]
        #alphas_1 = self.sqrt_alphas_cumprod[t_index].to(y.device)
        #alphas_2 = self.sqrt_one_minus_alphas_cumprod[t_index].to(y.device)
        return (alphas_1 * y + alphas_2 * noise)

    def diffusion_backward(self, x, z, inf_step=100, eta=0.5):
        if inf_step==self.time_step:
            """DDPM"""
            for t in reversed(range(0, self.time_step)):
                noise = torch.randn_like(z).to(x.device)
                t_tensor = torch.ones(size=[x.shape[0], 1]).to(x.device) * t / self.time_step
                pred_noise = self.model(x, z, t_tensor)
                z = self.sqrt_recip_alphas[t]*(z-self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * pred_noise) \
                    + torch.sqrt(self.posterior_variance[t]) * noise
        else: 
            """DDIM"""
            sample_time_step = torch.linspace(self.time_step-1, 0, (inf_step + 1)).to(x.device).to(torch.long)
            for i in range(1, inf_step + 1):
                t = sample_time_step[i - 1] 
                prev_t = sample_time_step[i] 
                noise = torch.randn_like(z).to(x.device)
                t_tensor = torch.ones(size=[x.shape[0], 1]).to(x.device) * t / self.time_step
                pred_noise = self.model(x, z, t_tensor)
                y_0 = (z - self.sqrt_one_minus_alphas_cumprod[t] * pred_noise) / self.sqrt_alphas_cumprod[t]
                var = eta * self.posterior_variance[t]
                z = self.sqrt_alphas_cumprod[prev_t] * y_0 + torch.sqrt(torch.clamp(1 - self.alphas_cumprod[prev_t] - var, 0, 1)) * pred_noise + torch.sqrt(var) * noise
        if self.normalize:
            return (z + 1) / 2
        else:
            return z

    def loss(self, noise_pred, noise):
        return self.criterion(noise_pred, noise)

class FM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(FM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.time_step = time_step
        self.min_sd = 0.01
        self.criterion = nn.L1Loss()
        self.normalize = output_norm

    def flow_forward(self, y, t, z, vec_type='gaussian'):
        if self.normalize:
            y = 2 * y - 1  # [0,1] normalize to [-1,1]
        if vec_type == 'gaussian':
            """
            t = 0:  N(0, 1)
            t = 1:  N(y, sd)
            Linear interpolation for mu and sigma
            """
            mu = y * t
            sigma = (self.min_sd) * t + 1 * (1 - t)
            noise = torch.randn_like(y).to(y.device)
            yt = mu + noise * sigma
            vec = (y - (1 - self.min_sd) * yt) / (1 - (1 - self.min_sd) * t)
        elif vec_type == 'conditional':
            mu = t * y + (1 - t) * z
            sigma = ((self.min_sd * t) ** 2 + 2 * self.min_sd * t * (1 - t)) ** 0.5
            noise = torch.randn_like(y).to(y.device)
            yt = mu + noise * sigma
            vec = (y - (1 - self.min_sd) * yt) / (1 - (1 - self.min_sd) * t)
        elif vec_type == 'rectified':
            """
            t = 0:  N(0, 1)
            t = 1:  N(y, sd)
            Linear interpolation for z and y
            """
            yt = t * y + (1 - t) * z
            vec = y-z
        elif vec_type == 'interpolation':
            """
            t = 0:  x
            t = 1:  N(0,1)
            Linear interpolation for z and y
            """
            # yt = (1 - t) * y + t * z
            yt = t * y + (1 - t) * z
            vec = None
            # return torch.cos(torch.pi/2*t) * y + torch.sin(torch.pi/2*t) * z
            # return (torch.cos(torch.pi*t) + 1)/2 * y + (torch.cos(-torch.pi*t) +1)/2  * z
        return yt, vec

    def flow_backward(self, x, z, step=0.01, method='Euler', direction='forward'):
        step = step if direction == 'forward' else -step
        t = 0 if direction == 'forward' else 1
        while (direction == 'forward' and t < 1) or (direction == 'backward' and t > 0):
            z += ode_step(self.model, x, z, t, step, method)
            t += step
        if self.normalize:
            return (z + 1) / 2
        else:
            return z

    def predict_vec(self, x, yt, t):
        vec_pred = self.model(x, yt, t)
        return vec_pred

    def loss(self, y, z, vec_pred, vec, vec_type='gaussian'):
        if vec_type in ['gaussian', 'rectified', 'conditional']:
            return self.criterion(vec_pred, vec)
        elif vec_type in ['interpolation']:
            loss = 1 / 2 * torch.sum(vec_pred ** 2, dim=1, keepdim=True) \
                   - torch.sum((y - z) * vec_pred, dim=1, keepdim=True)
            return loss.mean()
            # loss = 1/2 * torch.sum(vec **2, dim=-1, keepdim=True) - torch.sum((-torch.pi/2*torch.sin(torch.pi/2*t) * y +  torch.pi/2*torch.cos(torch.pi/2*t) * z) * vec, dim=-1, keepdim=True)
            # loss = 1/2 * torch.sum(vec **2, dim=-1, keepdim=True) - torch.sum((-torch.pi*torch.sin(torch.pi*t)*y +    torch.pi*torch.sin(-torch.pi*t) * z) * vec, dim=-1, keepdim=True)
        else:
            NotImplementedError


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def ode_step(model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, t: float, step: float, method: str = 'Euler'):
    model.eval()
    t_tensor = torch.ones(size=[x.shape[0], 1]).to(x.device) * t

    def model_eval(z_eval, t_eval):
        return model(x, z_eval, t_eval)

    with torch.no_grad():
        if method == 'Euler':
            v_pred = model_eval(z, t_tensor) * step
        else:
            v_pred_0 = model_eval(z, t_tensor) * step
            if method == 'Heun':
                v_pred_1 = model_eval(z + v_pred_0, t_tensor + step) * step
                v_pred = (v_pred_0 + v_pred_1) / 2
            elif method == 'Mid':
                v_pred = model_eval(z + v_pred_0 * 0.5, t_tensor + step * 0.5) * step
            elif method == 'RK4':
                v_pred_1 = model_eval(z + v_pred_0 * 0.5, t_tensor + step * 0.5) * step
                v_pred_2 = model_eval(z + v_pred_1 * 0.5, t_tensor + step * 0.5) * step
                v_pred_3 = model_eval(z + v_pred_2, t_tensor + step) * step
                v_pred = (v_pred_0 + 2 * v_pred_1 + 2 * v_pred_2 + v_pred_3) / 6
    return v_pred





class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1.0, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, x):
        gumbel_noise = self.sample_gumbel(x.size())
        y = x + gumbel_noise.to(x.device)
        soft_sample = F.softmax(y / self.temperature, dim=-1)

        if self.hard:
            hard_sample = torch.zeros_like(soft_sample).scatter(-1, soft_sample.argmax(dim=-1, keepdim=True), 1.0)
            sample = hard_sample - soft_sample.detach() + soft_sample
        else:
            sample = soft_sample

        return sample

    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(x)


class Time_emb(nn.Module):
    def __init__(self, emb_dim, time_steps, max_period):
        super(Time_emb, self).__init__()
        self.emb_dim = emb_dim
        self.time_steps = time_steps
        self.max_period = max_period

    def forward(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        t = t.view(-1) * self.time_steps
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.emb_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation, latent_dim=0, act='relu'):
        super(MLP, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        if latent_dim > 0:
            self.w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            input_dim = latent_dim
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), act)
        net = []
        for _ in range(num_layer):
            net.extend([nn.Linear(hidden_dim, hidden_dim), act])
            # net.append(ResBlock(hidden_dim, hidden_dim//4))
        net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)
        if output_activation:
            self.out_act = act_list[output_activation]
        else:
            self.out_act = nn.Identity()

    def forward(self, x, z=None, t=None):
        if z is None:
            emb = self.emb(x)
        else:
            emb = self.w(x) * self.emb(z) + self.b(x) #self.emb(torch.cat([x,z],dim=1))#
            if t is not None:
                emb = emb + self.temb(t)
        y = self.net(emb) 
        return self.out_act(y)


class Lip_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation, latent_dim=0):
        super(Lip_MLP, self).__init__()
        if latent_dim > 0:
            w = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
            b = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
            t = [Time_emb(hidden_dim, time_steps=1000, max_period=1000)]
            self.w = nn.Sequential(*w)
            self.b = nn.Sequential(*b)
            self.t = nn.Sequential(*t)
            net = []
        else:
            latent_dim = input_dim

        emb = [LinearNormalized(latent_dim, hidden_dim), nn.ReLU()]
        self.emb = nn.Sequential(*emb)
        net = []
        for _ in range(num_layer):
            net.extend([LinearNormalized(hidden_dim, hidden_dim), nn.ReLU()])
        net.append(LinearNormalized(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)

        if output_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif output_activation == 'gumbel':
            self.act = GumbelSoftmax(hard=True)
        elif output_activation == 'abs':
            self.act = Abs()
        else:
            self.act = nn.Identity()

    def forward(self, x, z=None, t=None):
        if z is None:
            emb = self.emb(x)
        else:
            emb = self.w(x) * self.emb(z) + self.b(x)
            if t is not None:
                emb = emb + self.t(t)
        y = self.net(emb)
        return self.act(y)

    def project_weights(self):
        self.net.project_weights()


class ATT(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation=None, latent_dim=0, agg=False,
                 pred_type='node', act='relu'):
        super(ATT, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        if latent_dim > 0:
            self.w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            input_dim = latent_dim
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), act)

        net = []
        # for _ in range(num_layer):
            # net.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            # net.extend([MHA(hidden_dim, 64, hidden_dim // 64),
            #             ResBlock(hidden_dim, hidden_dim//4)])
        # net.append(nn.Linear(hidden_dim, output_dim))
        # self.net = nn.Sequential(*net)
        self.net = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=max(hidden_dim // 64, 1))
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        # self.mha  = MHA(hidden_dim, 64, hidden_dim // 64)

        self.agg = agg
        self.pred_type = pred_type

        if output_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif output_activation == 'gumbel':
            self.act = GumbelSoftmax(hard=True)
        elif output_activation == 'abs':
            self.act = Abs()
        else:
            self.act = nn.Identity()

    def forward(self, x, z=None, t=None):
        ### x: B * N * F
        ### z: B * N
        ### t: B * 1
        batch_size = x.shape[0]
        node_size = x.shape[1]
        if z is None:
            # print(x.shape, self.emb)
            emb = self.emb(x)
        else:
            z = z.view(batch_size, -1, 1)
            emb = self.w(x) * self.emb(z) + self.b(x)
            if t is not None:
                emb = emb + self.temb(t).view(batch_size, 1, -1)
        emb = emb.permute(1, 0, 2)
        emb = self.trans(emb)
        emb = emb.permute(1, 0, 2)
        y = self.net(emb)  # B * N * 1
        
        if self.agg:
            y = y.mean(1)
        else:
            if self.pred_type == 'node':
                y = y.view(x.shape[0], -1)  # B * N
            else:
                y = torch.matmul(y.view(batch_size, node_size, 1), y.view(batch_size, 1, node_size))  # B * N * N
                col, row = torch.triu_indices(node_size, node_size, 1)
                y = y[:, col, row]
        return self.act(y)


class MHA(nn.Module):
    def __init__(self, n_in, n_emb, n_head):
        super().__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.key = nn.Linear(n_in, n_in)
        self.query = nn.Linear(n_in, n_in)
        self.value = nn.Linear(n_in, n_in)
        self.proj = nn.Linear(n_in, n_in)

    def forward(self, x):
        # x: B * node * n_in
        batch = x.shape[0]
        node = x.shape[1]
        ### softmax
        #### key: B H node emb
        #### que: B H emb node
        key = self.key(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,1,2)
        query = self.query(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,2,1)
        value = self.key(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,1,2)
        score = torch.matmul(key, query)/(self.n_emb**0.5) # x: B * H * node * node
        prob = torch.softmax(score, dim=-1) # B * H * node * node (prob)
        out = torch.matmul(prob, value) # B * H * Node * 64
        out = out.permute(0,2,3,1).contiguous() # B * N * F * H
        out = out.view(batch, -1, self.n_emb*self.n_head)
        return x + self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hid), 
                                 nn.ReLU(),
                                 nn.Linear(n_hid, n_in))
    def forward(self, x):
        return x + self.net(x)


class SDPBasedLipschitzLinearLayer(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6):
        super(SDPBasedLipschitzLinearLayer, self).__init__()

        self.activation = nn.ReLU(inplace=False)
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.rand(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        q_abs = torch.abs(self.q)
        q = q_abs[None, :]
        q_inv = (1 / (q_abs + self.epsilon))[:, None]
        T = 2 / (torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1) + self.epsilon)
        res = T * res
        res = F.linear(res, self.weights.t())
        out = x - res
        return out


class LinearNormalized(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearNormalized, self).__init__(in_features, out_features, bias)
        self.linear = spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)


class PartialLinearNormalized(nn.Module):
    def __init__(self, input_dim, output_dim, con_dim):
        super(PartialLinearNormalized, self).__init__()
        self.con_dim = con_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear_1 = spectral_norm(nn.Linear(input_dim - con_dim, output_dim))

    def forward(self, x):
        with torch.no_grad():
            weight_copy = self.linear_1.weight.data.clone()
            self.linear.weight.data[:, self.con_dim:] = weight_copy
        return self.linear(x)


class distance_estimator(nn.Module):
    def __init__(self, n_feats, n_hid_params, hidden_layers, n_projs=2, beta=0.5):
        super().__init__()
        self.hidden_layers = hidden_layers  # number of hidden layers in the network
        self.n_projs = n_projs  # number of projections to use for weights onto Steiffel manifold
        self.beta = beta  # scalar in (0,1) for stabilizing feed forward operations

        # Intialize initial, middle, and final layers
        self.fc_one = torch.nn.Linear(n_feats, n_hid_params, bias=True)
        self.fc_mid = nn.ModuleList(
            [torch.nn.Linear(n_hid_params, n_hid_params, bias=True) for i in range(self.hidden_layers)])
        self.fc_fin = torch.nn.Linear(n_hid_params, 1, bias=True)

        # Normalize weights (helps ensure stability with learning rate)
        self.fc_one.weight = nn.Parameter(self.fc_one.weight / torch.norm(self.fc_one.weight))
        for i in range(self.hidden_layers):
            self.fc_mid.weight = nn.Parameter(self.fc_mid[i].weight / torch.norm(self.fc_mid[i].weight))
        self.fc_fin.weight = nn.Parameter(self.fc_fin.weight / torch.norm(self.fc_fin.weight))

    def forward(self, u):
        u = self.fc_one(u).sort(1)[0]  # Apply first layer affine mapping
        for i in range(self.hidden_layers):  # Loop for each hidden layer
            u = u + self.beta * (self.fc_mid[i](u).sort(1)[0] - u)  # Convex combo of u and sort(W*u+b)
        u = self.fc_fin(u)  # Final layer is scalar (no need to sort)
        J = torch.abs(u)
        return J

    def project_weights(self):
        self.fc_one.weight.data = self.proj_Stiefel(self.fc_one.weight.data, self.n_projs)
        for i in range(self.hidden_layers):
            self.fc_mid[i].weight.data = self.proj_Stiefel(self.fc_mid[i].weight.data, self.n_projs)
        self.fc_fin.weight.data = self.proj_Stiefel(self.fc_fin.weight.data, self.n_projs)

    def proj_Stiefel(self, Ak, proj_iters):  # Project to closest orthonormal matrix
        n = Ak.shape[1]
        I = torch.eye(n)
        for k in range(proj_iters):
            Qk = I - Ak.permute(1, 0).matmul(Ak)
            Ak = Ak.matmul(I + 0.5 * Qk)
        return Ak



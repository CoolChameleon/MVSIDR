from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import numpy as np

class QueryLayer(nn.Module):
    def __init__(self, feature_dim=32, output_dim=128, depth_range=[425, 905], kernel_size=3, embed_fn=None, padding=24):
        super(QueryLayer, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.embed_fn = embed_fn
        self.kernel_size = kernel_size
        # assert self.kernel_size % 2 == 1, "Kernel_size must be an odd number"
        self.depth_range = depth_range
        self.depth_dim = 128
        self.padding = padding
        self.linear = nn.Linear(self.feature_dim, self.output_dim)

    def _get_center(self, point):
        p = point.clone().detach()
        if self.padding:
            p[0], p[1] = p[0] + self.padding, p[1] + self.padding
        p[2] = (p[2] - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0]) * self.depth_dim

        center = (np.floor(p) + np.ceil(p)) / 2

        return center

    def attention(self, query, key, value):
        q = query.clone().detach()
        if self.padding:
            q[0], q[1] = q[0] + self.padding, q[1] + self.padding
        q[2] = (q[2] - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0]) * self.depth_dim

        q = q.unsqueeze(0)
        k = torch.stack([torch.tensor(i, dtype=torch.float32) for i in list(key)])
        v = torch.stack(list(value))
        # print(q.shape, k.shape, v.shape)
        sim_map = torch.matmul(k, q.transpose(0, 1))
        sim_map = sim_map / torch.sum(sim_map)
        mixed_feature = torch.matmul(sim_map.transpose(0, 1), v)
        return mixed_feature

    def forward(self, sampled_points, feature_volume):
        # sampled_points: [B * 3] [v, u, d]像素坐标系的uv还有d，来自深度图。也可以用ray_tracing输出转化
        # feature_volume: [h * w * d * c]
        # output: [B * ]

        output = []
        
        for point in sampled_points:
            center = self._get_center(point)
            query_pool = dict()
            for i in range(int(center[0] - self.kernel_size * 0.5), int(center[0] + self.kernel_size * 0.5) + 1):
                for j in range(int(center[1] - self.kernel_size * 0.5), int(center[1] + self.kernel_size * 0.5) + 1):
                    for k in range(int(center[2] - self.kernel_size * 0.5), int(center[2] + self.kernel_size * 0.5) + 1):
                        query_pool[(i, j, k)] = feature_volume[i, j, k]
            
            # print("point coord =", point)
            point_feature = self.attention(point, query_pool.keys(), query_pool.values())
            point_feature = point_feature / torch.sum(point_feature)
            # print("point_feature =", point_feature)
            point_bias = torch.squeeze(self.linear(point_feature))
            # print("point_bias =", point_bias)
            output.append(point_bias)        
        
        return torch.stack(output)

class QueryDecoder(nn.Module):
    def __init__(self, output_dim=257, num_layers=8, skip_in=[4], latent_dim=128, embed_fn=None):
        super(QueryDecoder, self).__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.skip_in = skip_in
        self.latent_dim = latent_dim
        self.embed_fn = embed_fn    
        
        if self.embed_fn:
            self.imput_dim = embed_fn.get_dim()
        else:
            self.input_dim = 3
        
        self.linear_layers = []  
        self.linear_layers.append(nn.Linear(self.input_dim, self.latent_dim))
        for i in range(self.num_layers):
            # if i not in skip_in:
            #     self.linear_layers.append(nn.Linear(self.latent_dim, self.latent_dim))
            # else:
            #     self.linear_layers.append(nn.Linear(self.latent_dim + self.input_dim, self.latent_dim))
            if i + 1 in self.skip_in:
                self.linear_layers.append(nn.Linear(self.latent_dim, self.latent_dim - self.input_dim))
            else:
                self.linear_layers.append(nn.Linear(self.latent_dim, self.latent_dim))

        self.linear_layers.append(nn.Linear(self.latent_dim, self.output_dim))
        self.query = QueryLayer()
        self.softplus = nn.Softplus(beta=100)

    def forward(self, sampled_points_world, sampled_points_pixel, feature_volume):
        if self.embed_fn:
            sampled_points_world = self.embed_fn(sampled_points_world)
        points_bias = self.query(sampled_points_pixel, feature_volume)
        points_bias = points_bias / torch.mean(torch.abs(points_bias))
        # print("mean points_bias =", torch.mean(torch.abs(points_bias), dim=1))

        x = sampled_points_world
        
        for l in range(0, self.num_layers + 2):    

            x = self.linear_layers[l](x)
            if l in self.skip_in:
                # x = torch.cat([x, sampled_points_world], 1) / np.sqrt(2)
                x = torch.cat([x, sampled_points_world], 1)
            # print(l, x.shape)
            if l in [0]:
                x = x * points_bias

            if l < self.num_layers + 1:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

        # TODO

if __name__ == "__main__":
    print("Hi!")
    # q = QueryLayer(output_dim=128)
    sampled_points_world = torch.tensor(np.array([[ 34.91919707, 74.78540685, 502.22212776], [127.31929888, 37.90384561, 506.21172182], [ 14.00954739, 18.73507563, 464.03965051], [129.14810869, -2.4058813, 492.47852358], [ 83.75214084,  8.50605099, 471.42095505], [ 90.81179813, 77.27596583, 509.24331148], [160.84538864, -20.48992874, 511.16228426], [164.3325585, 10.01046642, 523.75870379], [125.94592973, 57.87146125, 586.32306453], [144.43098415, -17.65466897, 497.03256245]]), dtype=torch.float32)

    sampled_points_pixel = torch.tensor([[18, 87, 15, 98, 60, 51, 129, 125, 100, 114], [67, 57, 17, 27, 19, 78, 29, 53, 104, 22], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32)

    # depth = torch.tensor([[492.73355, 461.5603, 503.65237, 469.63785, 475.02255, 465.61923, 473.47656, 466.2083, 508.71964, 471.4123]], dtype=torch.float32)
    depth = torch.tensor([[492.73355, 461.5603, 503.65237, 469.63785, 475.02255, 465.61923, 473.47656, 466.2083, 508.71964, 471.4123]], dtype=torch.float32)

    sampled_points_pixel = torch.cat((sampled_points_pixel[:2], depth), dim=0).permute(1, 0)

    feature_volume = torch.tensor(np.load("/ceph/home/lsp20/SHJ/mvsnerf/volume.npy"))
    
    feature_volume = torch.squeeze(feature_volume).permute(2, 3, 1, 0)
    # ql = QueryLayer()
    # ql(sampled_points_pixel, feature_volume)
    # print(ql(sampled_points, feature_volume).shape)
    qd = QueryDecoder()
    output = qd(sampled_points_world, sampled_points_pixel, feature_volume)
    print("OUTPUT SHAPE =", output.shape)
    print(output)

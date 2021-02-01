import numpy as np
import torch
from torch_geometric.data import Data, Batch


from mcmctools.pytorch.data_generation.batchconfigdatagenerator import BatchConfigDataGenerator


class BatchGraphDataGenerator(BatchConfigDataGenerator):
    def __init__(self, **kwargs):
        kwargs["data_type"] = "target_param"
        super().__init__(**kwargs)

        self.dimensions = kwargs.pop("dimensions")
        self.config_size = np.prod(self.dimensions)
        self.self_loops = True
        edge_type = kwargs.pop("edge_type", "nearest_neighbour")
        self.edge_indices = self.generate_edge_indices(edge_type)

    def get_config_dimensions(self):
        return self.dimensions

    def sample_target_param(self):
        config, target = super().sample_target_param()
        config = torch.tensor(config, dtype=torch.float32, device=self.device).view(len(config), self.config_size, -1)
        config = Batch.from_data_list([Data(x=conf, edge_index=self.edge_indices, y=torch.tensor([tar], dtype=torch.float32, device=self.device)) for (conf, tar) in zip(config, target)])
        return config

    def generate_edge_indices(self, edge_type):
        if edge_type == "nearest_neighbour":
            self.generate_nearest_neighbour_edge_indices()
        elif edge_type == "plaquette":
            self.generate_plaquette_edge_indices()
        else:
            assert False, "Edge type not known."

    def generate_nearest_neighbour_edge_indices(self):
        from mcmctools.utils.lattice import get_neighbour_index
        dim_mul = np.cumprod([1] + self.dimensions)
        elem_per_site = 1

        edge_indices = []
        for i in range(self.config_size):

            for dim in range(len(self.dimensions)):
                edge_indices.append([i, get_neighbour_index(n=i, dim=dim, direction=True, mu=0, dim_mul=dim_mul,
                                                            dimensions=self.dimensions, elem_per_site=elem_per_site)])
                edge_indices.append([i, get_neighbour_index(n=i, dim=dim, direction=False, mu=0, dim_mul=dim_mul,
                                                            dimensions=self.dimensions,  elem_per_site=elem_per_site)])

            if self.self_loops:
                edge_indices.append([i, i])
        edge_indices = np.array(edge_indices).T
        return torch.tensor(edge_indices, dtype=torch.long, device=self.device)

    def generate_plaquette_edge_indices(self):
        assert False, "Plaquette edge indices not implemented"

    def convert_to_config_data(self, batch):
        config = batch.x.view(-1, np.prod(self.dimensions))
        beta = batch.y.to(config.device)
        return config, beta

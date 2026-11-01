from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
import torchvision.transforms.functional as F
import rasterio
import os


OAM_NORM_MEAN = [0.4519094796444375, 0.46631762029216894, 0.36423604965610323]
OAM_NORM_STDV = [0.23539068291108825, 0.2267043341867209, 0.2203186645671427]


def collate_fn(batch):
    return tuple(zip(*batch))


class MinimalPredictTifDataset(Dataset):
    def __init__(self, data_path, oam_normalize=True):
        self.data_path = data_path
        self.predict_tile_list = os.listdir(data_path)
        self.predict_tile_list = [tile for tile in self.predict_tile_list if tile.endswith(".tif")]
        self.oam_normalize = oam_normalize
        self.normalize = Normalize(mean=OAM_NORM_MEAN, std=OAM_NORM_STDV)

    def __len__(self):
        return len(self.predict_tile_list)

    def __getitem__(self, idx):
        tile_name = self.predict_tile_list[idx]
        # tile = imread(os.path.join(self.data_path, tile_name))
        with rasterio.open(os.path.join(self.data_path, tile_name)) as src:
            transform = src.transform
            crs = src.crs
            tile = src.read()
        if tile.shape[0] == 3:
            tile = F.to_tensor(tile).permute(1, 0, 2)
        else:
            tile = F.to_tensor(tile)
        if self.oam_normalize:
            tile = self.normalize(tile)
        metadata = {
            "tile_name": tile_name,
            "transform": transform,
            "crs": crs
        }
        return tile, metadata


def create_predict_loader(data_path, batch_size, num_workers, oam_normalization=False):
    """
    Create dataloader from the prediction data directory
    :param data_path: full/path/to/prediction/data/directory/
    :return: dataloader
    """
    dataset = MinimalPredictTifDataset(data_path, oam_normalization)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
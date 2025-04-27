import os

import torch
import yaml
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets

from data.dali_pipeline import simclr_dali_pipeline
from data.dali_iterator import DALIGenericIteratorWithViews
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer

print(torch.__version__)
torch.manual_seed(0)


def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    if config['dali']['enabled']:
        dali_pipeline = simclr_dali_pipeline(config['dataset']['root'], batch_size=config['trainer']['batch_size'],
                                           num_threads=config['trainer']['num_workers'], device_id=0, reader_name=config['dali']['reader_name'],
                                           image_size=eval(config['data_transforms']['input_shape'])[0], shuffle=True)
        dali_pipeline.build()
        train_loader = DALIGenericIteratorWithViews(dali_pipeline, ['view1', 'view2', "label"], reader_name=config['dali']['reader_name'])
    else:
        data_transform = get_simclr_data_transforms(**config['data_transforms'])
        train_dataset = datasets.STL10('/home/thalles/Downloads/', split='train+unlabeled', download=True,
                                   transform=MultiViewDataInjector([data_transform, data_transform]))
        train_loader = DataLoader(train_dataset, batch_size=config['trainer']['batch_size'],
                                num_workers=config['trainer']['num_workers'], drop_last=False, shuffle=True)
    
    
    # online network
    online_network = ResNet18(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_loader)


if __name__ == '__main__':
    main()

import os
import argparse
import json
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from dataset import LUMINA_Density, MultiImagesDataset
from model import get_model
from train import test_fn
import matplotlib.pyplot as plt
import numpy as np

def load_data(args):
    test_ds = []

    transform_test = v2.Compose([
        # v2.ToImage(), 
        v2.Resize((args.input_size, args.input_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=3),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
            
    test_data = LUMINA_Density(root = os.path.join(args.data_path, 'LUMINA_PNG'))
    test_ds = MultiImagesDataset(test_data, transform=transform_test)
    print(f"There are {len(test_data['label'])} testing samples.")

    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return test_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MammoFL")
    parser.add_argument("--data-path", default="/dataset/Mammogram", type=str, help="dataset path")
    parser.add_argument("--model", default="swin_t", type=str, help="model name")
    parser.add_argument("-o", "--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("-r", "--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of train data loading workers")
    parser.add_argument("--input-size", default=224, type=int, help="input size")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model+'_'+str(args.input_size))
        
    device = torch.device(args.device)
    test_dataloader = load_data(args)

    model = get_model(name = args.model, num_classes = 4)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.resume), map_location='cpu', weights_only=True))
    
    epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)
    print(f"Test loss {epoch_log['loss']:.4f} acc {epoch_log['acc']:.4f} auc {epoch_log['auc']:.4f} f1 {epoch_log['f1']:.4f}")
    print(f"Latex: {epoch_log['acc']*100:.2f} & {epoch_log['auc']*100:.2f} & {epoch_log['f1']*100:.2f}")
    
    with open(os.path.join(args.output_dir, "log.json")) as json_file:
        log_hist = json.load(json_file)
    
    for metric in ['loss', 'acc']:
        plt.figure(figsize=(8,6))
        plt.plot(np.arange(1, 1+len(log_hist['test_'+metric])), log_hist['train_'+metric], lw=2, label='Train')
        plt.plot(np.arange(1, 1+len(log_hist['test_'+metric])), log_hist['test_'+metric], lw=2, label='Test')
        plt.xlim([0, len(log_hist['test_'+metric])])
        #plt.ylim([0, 1])
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args.output_dir, metric+".pdf"), format="pdf", bbox_inches='tight')
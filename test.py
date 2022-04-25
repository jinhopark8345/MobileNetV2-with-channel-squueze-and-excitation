import os
import numpy as np
import datetime
import argparse
import torch
from pathlib import Path

import utils
from model import MobileNetV2_with_CSE
from data_loader import CIFAR10DataLoader


class Tester:
    def __init__(
        self,
        model,
        test_data_loader,
        device,
    ):
        self.model = model
        self.test_data_loader = test_data_loader
        self.device = device

        classes = ["airplane" , "automobile" , "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.idx2class = {idx : cls for idx, cls in enumerate(classes)}

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, targets) in self.test_data_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))

        class_correct = list(0.0 for _ in range(10))
        class_total = list(0.0 for _ in range(10))

        with torch.no_grad():
            for (images, targets) in self.test_data_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == targets).squeeze()
                for batch_idx in range(4):
                    label = targets[batch_idx]
                    class_correct[label] += c[batch_idx].item()
                    class_total[label] += 1

        for batch_idx in range(10):
            print(
                "Accuracy of %5s : %2d %%" % (
                    self.idx2class[batch_idx],
                    100 * class_correct[batch_idx] / class_total[batch_idx],
                )
            )

def main(data_dir, saved_model_path, batch_size, summary):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_data_loader = CIFAR10DataLoader(data_dir, batch_size=batch_size, training=False)

    # bottleneck blocks configurations:
    #   (expansion, out_planes, num_blocks, stride)
    cfg_btn = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 1),
    ]

    # channel-wise squeeze and excitation configurations:
    #   (out_planes, num_blocks, stride)
    cfg_cse = [
        (64, 4, 2),
        (96, 3, 2),
        (160, 3, 1),
        (320, 1, 2),
    ]

    model = MobileNetV2_with_CSE(cfg_btn, cfg_cse).to(device)
    model.load_state_dict(torch.load(saved_model_path)["model"])


    tester = Tester(model, test_data_loader=test_data_loader, device=device)
    tester.test()



if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="PyTorch MobileNetV2 based model CIFAR10 Testing"
    )
    args.add_argument("--data_dir", type=utils.is_dir_path)
    args.add_argument("--saved_model", type=utils.is_file_path)
    args.add_argument("--batch_size", default=16, type=int)
    args.add_argument("--summary", default=True, type=bool)

    args = args.parse_args()
    main(args.data_dir, args.saved_model, args.batch_size, args.summary)

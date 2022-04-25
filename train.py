import os
import numpy as np
import datetime
import argparse
import torch

import utils
from pathlib import Path

from model import MobileNetV2_with_CSE
from data_loader import CIFAR10DataLoader


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        train_data_loader,
        validation_data_loader,
        best_acc,
        start_epoch,
        end_epoch,
        report_interval,
        save_folder_path,
        save_start_epoch=20,
    ):

        self.model = model

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.device = device

        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader

        self.best_acc = best_acc
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.report_interval = report_interval

        self.save_folder_path = save_folder_path
        self.save_start_epoch = save_start_epoch

    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            train_loss = 0.0
            for (inputs, labels) in self.train_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                # train_loss += loss.item()
                # print(f'{epoch, batch_idx + 1, train_loss = }')
                # if (batch_idx + 1) % int(self.report_interval / batch_size) == 0:  # print every 2000 mini-batches
                #     print(
                #         "[%d, %5d] loss: %.9f"
                #         % (epoch, batch_idx + 1, train_loss / 2000)
                #     )
                #     train_loss = 0.0

            with torch.no_grad():
                if epoch % 1 == 0:
                    correct = 0
                    total = 0
                    for (images, labels) in self.validation_data_loader:
                        images, labels = images.to(self.device), labels.to(
                            self.device
                        )
                        outputs = self.model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                cur_acc = 100 * correct / total

                if cur_acc > self.best_acc:
                    self.best_acc = cur_acc
                    print(
                        f"Accuracy of the network on the 10000 test images: {self.best_acc}"
                    )

                    if epoch >= self.save_start_epoch:
                        acc_without_dot = f"{self.best_acc:.2f}".replace(
                            ".", ""
                        )
                        save_path = (
                            self.save_folder_path
                            + f"epoch-{epoch}"
                            + f"-acc-{acc_without_dot}.pth"
                        )
                        checkpoint = {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "lr": self.lr_scheduler.state_dict(),
                            "acc": self.best_acc,
                            "loss": loss,
                            "epoch": epoch,
                        }
                        torch.save(
                            checkpoint,
                            save_path,
                        )

            # scheduler
            self.lr_scheduler.step()


def resume_training(model, optimizer, checkpoint_path):
    print("==> Resuming from checkpoint..")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduled.load_state_dict(checkpoint["lr"])
    best_acc = checkpoint["acc"]
    # because if we resume, it will start training epoch should be +1 from now
    start_epoch = checkpoint["epoch"] + 1
    return model, optimizer, lr_scheduled, best_acc, start_epoch


def make_save_folder(save_folder_path):
    try:
        os.makedirs(save_folder_path)
    except OSError:
        print("Creation of the directory %s failed" % save_folder_path)

    print(f"save_folder_path: {save_folder_path}")


def main(data_dir, epoch, batch_size, summary, resume=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = CIFAR10DataLoader(
        data_dir, batch_size=batch_size, training=True
    )
    validation_loader = CIFAR10DataLoader(
        data_dir, batch_size=batch_size, training=False
    )

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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.045, momentum=0.9, weight_decay=4e-5
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.98)
    report_interval = 25000
    best_acc = 0
    start_epoch = 0
    resume_checkpoint_path = ""

    if resume:
        resume_checkpoint_path = args.resume
        model, optimizer, lr_scheduled, best_acc, start_epoch = resume_training(
            model,
            optimizer,
            resume_checkpoint_path,
        )
        save_folder_path = str(Path(resume_checkpoint_path).parent) + "/"
    else:
        save_folder_path = (
            "./save/"
            + datetime.datetime.now().strftime("%Y-%m-%d-%Hh-%Mm-%Ss")
            + f"-batch_size-{batch_size}-max_epoch-{epoch}/"
        )

        make_save_folder(save_folder_path)

    if summary:
        import torchsummary
        torchsummary.summary(model, (3, 32, 32), device=device)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_data_loader=train_loader,
        validation_data_loader=validation_loader,
        lr_scheduler=lr_scheduler,
        best_acc=0,
        start_epoch=start_epoch,
        end_epoch=epoch,
        report_interval=report_interval,
        save_folder_path=save_folder_path,
        save_start_epoch=1,
    )
    trainer.train()

    print("Finished Training")


if __name__ == "__main__":

    args = argparse.ArgumentParser(
        description="PyTorch MobileNetV2 based model CIFAR10 Training"
    )
    args.add_argument("--data_dir", type=utils.is_dir_path)
    args.add_argument("--epochs", default=200, type=int)
    args.add_argument("--batch_size", default=16, type=int)
    args.add_argument("--summary", default=True, type=bool)
    args.add_argument(
        "--resume",
        default=None,
        type=lambda x: utils.is_valid_file(parser, x),
        help="path to latest checkpoint (default: None)",
    )

    args = args.parse_args()
    main(args.data_dir, args.epochs, args.batch_size, args.summary, args.resume)

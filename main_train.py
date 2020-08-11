import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_set import ISONetData
from isonet import ISONet
from utils import time2str


def main_train(args):
    # Get the args
    if args.resume_training is not None:
        if not os.path.isfile(args.resume_training):
            print(f"{args.resume_training} not a valid file!")
            return
        else:
            print(f"load checkpoint:{args.resume_training}")
    cuda = args.cuda
    resume = args.resume_training
    batch_size = args.batch_size
    milestones = args.milestones
    lr = args.lr
    total_epoch = args.epochs
    resume_checkpoint_filename = args.resume_training
    best_model_name = args.best_model_name
    checkpoint_name = args.best_model_name
    data_path = args.data_path
    start_epoch = 1

    print("Loading data....")
    dataset = ISONetData(data_path=data_path)
    dataset_test = ISONetData(data_path=data_path, train=False)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
    print("Loading data success...")
    print(f"Size of training data: {len(dataset)}")
    print(f"Size of validation data: {len(dataset_test)}")

    model_path = Path("models")
    checkpoint_path = model_path.joinpath("checkpoint")

    if not model_path.exists():
        model_path.mkdir()
    if not checkpoint_path.exists():
        checkpoint_path.mkdir()

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        print("cuda unavailable!")
        cuda = False

    net = ISONet()
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(net.parameters(), lr=lr)

    if cuda:
        net = net.to(device=device)
        criterion = criterion.to(device=device)

    scheduler = MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
    writer = SummaryWriter()

    # Resume training
    if resume:
        print("Resume training...")
        checkpoint = torch.load(checkpoint_path.joinpath(resume_checkpoint_filename))
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict((checkpoint["optimizer"]))
        scheduler.load_state_dict(checkpoint["scheduler"])
        resume_epoch = checkpoint["epoch"]
        best_test_loss = checkpoint["best_test_loss"]

        start_epoch = resume_epoch + 1
        print(f"start from the [{start_epoch}]th epoch...")
        print(f"Loss of the last epoch: [{best_test_loss}]...")
    else:
        # Initialize the weights
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    if not locals().get("best_test_loss"):
        best_test_loss = 0

    record = 0
    for epoch in range(start_epoch, total_epoch):
        print(f"start from [{epoch}] epoch...")
        net.train()
        writer.add_scalar("Train/Learning Rate", scheduler.get_last_lr()[0], epoch)
        for i, (data, label) in enumerate(data_loader, 0):
            if i == 0:
                start_time = int(time.time())
            if cuda:
                data = data.to(device=device)
                label = label.to(device=device)
            label = label.unsqueeze(1)

            optimizer.zero_grad()

            output = net(data)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()
            if i % 500 == 499:
                end_time = int(time.time())
                use_time = end_time - start_time

                print(
                    f">>> epoch[{epoch}] loss[{loss:.4f}]  {i * batch_size}/{len(dataset)} lr{scheduler.get_last_lr()} ",
                    end="")
                left_time = ((len(dataset)-i*batch_size)/500/batch_size)*(end_time-start_time)
                print(f"Running time: [{end_time - start_time:.2f}]seconds, remaining time estimated: [{left_time:.2f}] seconds")
                start_time = end_time
            # record to tensorboard
            if i % 128 == 127:
                writer.add_scalar("Train/loss", loss, record)
                record += 1

        # validate
        print("Validate the model...")
        net.eval()

        test_loss = 0
        with torch.no_grad():
            loss_t = nn.MSELoss(reduction="mean")
            if cuda:
                loss_t = loss_t.to(device)
            for data, label in data_loader_test:
                if cuda:
                    data = data.to(device)
                    label = label.to(device)
                # expand dim
                label = label.unsqueeze_(1)
                predict = net(data)
                # sum up batch loss
                test_loss += loss_t(predict, label).item()

        test_loss /= len(dataset_test)
        test_loss *= batch_size
        print(f'\nTest Data: Average batch[{batch_size}] loss: {test_loss:.4f}\n')
        scheduler.step()

        writer.add_scalar("Test/Loss", test_loss, epoch)

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scheduler": scheduler.state_dict(),
            "best_test_loss": best_test_loss
        }

        if best_test_loss == 0:
            print("Save the model...")
            torch.save(net.state_dict(), model_path.joinpath(best_model_name))
            best_test_loss = test_loss
        else:
            # save a better model
            if test_loss < best_test_loss:
                print("a better model is available, saving...")
                torch.save(net.state_dict(), model_path.joinpath(best_model_name))
                best_test_loss = test_loss
        # save the checkpoint
        if epoch % args.save_every_epochs == 0:
            c_time = time2str()
            torch.save(checkpoint, checkpoint_path.joinpath(
                f"{checkpoint_name}_{epoch}_{c_time}.cpth"))
            print(f"save the checkpoint: [{checkpoint_name}_{epoch}_{c_time}.cpth]...\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--cuda", type=bool, default=True, help="whether to use cuda")
    parser.add_argument("--milestones", type=int, default=[10, 30], nargs=2,
                        help="when to change learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--best_model_name", type=str, default="net.pth", help="Model_name")
    parser.add_argument("--data_path", type=str, default="data_64_64_aug3",
                        help="directory_of_training_set")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial_learning_rate")
    parser.add_argument("--resume_training", type=str, help="whether_to_resume")
    parser.add_argument("--save_every_epochs", type=int, default=1, help="frequency_of_save_checkpoint")

    args = parser.parse_args()
    # Print parameters
    for p, v in args.__dict__.items():
        print('\t{}: {}'.format(p, v))
    main_train(args=args)

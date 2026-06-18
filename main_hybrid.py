import argparse
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from core_scripts.startup_config import set_random_seed
from data_utils_SSL import (
    Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2021_eval,
    Dataset_in_the_wild_eval,
    genSpoof_list,
)
from model_hybrid import Model


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def default_eval_output(track):
    name = "scores_Hybrid_Wild.txt" if track == "In-the-Wild" else "scores_Hybrid_{}.txt".format(track)
    return os.path.join("scores", name)


def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            batch_loss = criterion(batch_out, batch_y)
            val_loss += batch_loss.item() * batch_size

    val_loss /= num_total
    acc = 100 * (num_correct / num_total)
    return val_loss, acc


def produce_evaluation_file(dataset, model, device, save_path, batch_size=8):
    if save_path is None:
        raise ValueError("--eval_output is required in evaluation mode")
    ensure_parent_dir(save_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()

    with open(save_path, "w") as fh:
        with torch.no_grad():
            for batch_x, utt_id in tqdm(data_loader):
                batch_x = batch_x.to(device)
                batch_out = model(batch_x)
                batch_score = batch_out[:, 1].data.cpu().numpy().ravel()
                for f, cm in zip(utt_id, batch_score.tolist()):
                    fh.write("{} {}\n".format(f, cm))
    print("Scores saved to {}".format(save_path))


def train_epoch(train_loader, model, optimizer, device):
    running_loss = 0.0
    num_total = 0.0
    model.train()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in tqdm(train_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return running_loss / num_total


def add_rawboost_args(parser):
    parser.add_argument("--algo", type=int, default=3)
    parser.add_argument("--nBands", type=int, default=5)
    parser.add_argument("--minF", type=int, default=20)
    parser.add_argument("--maxF", type=int, default=8000)
    parser.add_argument("--minBW", type=int, default=100)
    parser.add_argument("--maxBW", type=int, default=1000)
    parser.add_argument("--minCoeff", type=int, default=10)
    parser.add_argument("--maxCoeff", type=int, default=100)
    parser.add_argument("--minG", type=int, default=0)
    parser.add_argument("--maxG", type=int, default=0)
    parser.add_argument("--minBiasLinNonLin", type=int, default=5)
    parser.add_argument("--maxBiasLinNonLin", type=int, default=20)
    parser.add_argument("--N_f", type=int, default=5)
    parser.add_argument("--P", type=int, default=10)
    parser.add_argument("--g_sd", type=int, default=2)
    parser.add_argument("--SNRmin", type=int, default=10)
    parser.add_argument("--SNRmax", type=int, default=40)


def build_parser():
    parser = argparse.ArgumentParser(description="Hybrid SLS experimental system")
    parser.add_argument("--database_path", type=str, default="/path/to/your/database/")
    parser.add_argument("--protocols_path", type=str, default="database/")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--loss", type=str, default="weighted_CCE")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--xlsr_checkpoint", type=str, default="xlsr2_300m.pt")
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--track", type=str, default="DF", choices=["LA", "In-the-Wild", "DF"])
    parser.add_argument("--eval_output", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--is_eval", action="store_true", default=False)
    parser.add_argument("--eval_part", type=int, default=0)
    parser.add_argument("--disable_cudnn", action="store_true", default=False)

    parser.add_argument("--use_stat_sls", type=int, default=1, choices=[0, 1])
    parser.add_argument("--stat_sls_use_std", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_swiglu", type=int, default=1, choices=[0, 1])
    parser.add_argument("--pooling_type", type=str, default="cgta", choices=["maxpool", "temporal", "cgta"])
    parser.add_argument("--cgta_use_std", type=int, default=1, choices=[0, 1])
    parser.add_argument("--cgta_stat_residual", type=int, default=1, choices=[0, 1])
    parser.add_argument("--hybrid_hidden_dim", type=int, default=128)
    parser.add_argument("--hybrid_dropout", type=float, default=0.1)
    parser.add_argument("--load_strict", type=int, default=1, choices=[0, 1])
    parser.add_argument("--early_stop_patience", type=int, default=3)

    add_rawboost_args(parser)
    return parser


def load_checkpoint(model, args, device):
    if not args.model_path:
        return
    state_dict = torch.load(args.model_path, map_location="cpu")
    if bool(args.load_strict):
        model.load_state_dict(state_dict)
        print("Model loaded : {}".format(args.model_path))
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Model partially loaded : {}".format(args.model_path))
        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))
    del state_dict
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.eval and args.eval_output is None:
        args.eval_output = default_eval_output(args.track)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cudnn enabled: False")

    os.makedirs("models", exist_ok=True)
    set_random_seed(args.seed, args)

    model_tag = "hybrid_{}_{}_{}_{}_stat{}_swiglu{}_pool{}".format(
        args.track,
        args.loss,
        args.num_epochs,
        args.batch_size,
        args.use_stat_sls,
        args.use_swiglu,
        args.pooling_type,
    )
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_save_path = os.path.join("models", model_tag)
    os.makedirs(model_save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    model = Model(args, device)
    nb_params = sum(param.view(-1).size()[0] for param in model.parameters())
    model = nn.DataParallel(model).to(device)
    print("nb_params:", nb_params)
    print(
        "hybrid config: stat_sls={} swiglu={} pooling={} hidden_dim={}".format(
            args.use_stat_sls,
            args.use_swiglu,
            args.pooling_type,
            args.hybrid_hidden_dim,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    load_checkpoint(model, args, device)

    if args.track == "In-the-Wild":
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print("no. of eval trials", len(file_eval))
        eval_set = Dataset_in_the_wild_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output, args.eval_batch_size)
        sys.exit(0)

    if args.eval:
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print("no. of eval trials", len(file_eval))
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output, args.eval_batch_size)
        sys.exit(0)

    train_protocol = os.path.join(
        args.protocols_path,
        "ASVspoof_DF_cm_protocols",
        "ASVspoof2019.LA.cm.train.trn.txt",
    )
    d_label_trn, file_train = genSpoof_list(dir_meta=train_protocol, is_train=True, is_eval=False)
    print("no. of training trials", len(file_train))
    train_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(args.database_path, "ASVspoof2019_LA_train"),
        algo=args.algo,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )

    writer = SummaryWriter("logs/{}".format(model_tag))
    best_train_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.num_epochs):
        running_loss = train_epoch(train_loader, model, optimizer, device)
        writer.add_scalar("loss", running_loss, epoch)
        print("\n{} - train_loss={}".format(epoch, running_loss))

        epoch_path = os.path.join(model_save_path, "epoch_{}.pth".format(epoch))
        torch.save(model.state_dict(), epoch_path)

        if running_loss < best_train_loss:
            best_train_loss = running_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, "best.pth"))
        else:
            patience_counter += 1

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print("Early stopping triggered. best_train_loss={}".format(best_train_loss))
            break


if __name__ == "__main__":
    main()

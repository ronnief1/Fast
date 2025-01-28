import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse
import datetime
import numpy as np
import pandas as pd
import time
from csv import writer
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy

from src.retfound.vit import vit_large_patch16
from src.multimae.v2_main import MultiMAEWrapper
from src.multimae.v1_main import MultiMAEWrapper as MultiMAEWrapperv1
from src.utils.dataset import build_dataset
from src.retfound.pos_embed import interpolate_pos_embed
import src.retfound.lr_decay as lrd
import src.retfound.misc as misc
from src.utils.train_eval import *

import warnings

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Retinal image classification experiments", add_help=False
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--input_size",
        default=224,
        type=int,
        help="Images input size: 224 for Random, ImageNet, RetFound and 512 for MultiMAE.",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-8,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Label smoothing (default: 0.1). If 0 -> CrossEntropyLoss.",
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints).",
    )

    # * Supervised training params
    parser.add_argument(
        "--linear_probing",
        default=False,
        type=bool,
        help="Set to True for not training the encoder weights.",
    )
    parser.add_argument(
        "--label_efficiency_exp",
        default=True,
        type=bool,
        help="Set to True for label efficiency experiments.",
    )
    parser.add_argument(
        "--init_weights",
        default="/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp/RETFound_oct_weights.pth",
        type=str,
        help="Pre-trained weights to initialise the model with. Default: RetFound.",
    )
    parser.add_argument(
        "--resume", default="", help="Resume from checkpoint - for supervised training."
    )
    parser.add_argument("--task", default="", type=str, help="name of the experiment")
    parser.add_argument(
        "--global_pool",
        default=True,
        type=bool,
        help="By default we're using global pooling instead of the CLS token for classifier input.",
    )
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument(
        "--output_dir",
        default="/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp/",
        help="path where to save, empty for no saving",
    )

    # * Dataset parameters
    parser.add_argument(
        "--data_root",
        default="/msc/home/esueck47/data/foundoptima_clf_data/",
        type=str,
        help="dataset path",
    )
    parser.add_argument("--data_set", default="", type=str, help="dataset folder name")
    parser.add_argument(
        "--imgnet_scaler",
        default=True,
        type=bool,
        help="Whether to normalize with ImageNet mean and std (like in RetFound)). Set to False for MultiMAE experiments.",
    )
    parser.add_argument(
        "--nb_classes", default=2, type=int, help="Number of the classification types"
    )
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # * Training parameters
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument(
        "--eval",
        default=False,
        type=bool,
        help="Set to True for only running the evaluation on the test set.",
    )
    parser.set_defaults(eval=False)
    parser.add_argument(
        "--early_stopping_epochs",
        default=200,
        type=int,
        help="Parameter to control how many epochs to wait for the validation loss to improve before stopping.",
    )
    parser.add_argument(
        "--early_stopping_delta",
        default=0.001,
        type=float,
        help="Parameter to specify the minimum change in the validation loss required to consider it an improvement.",
    )

    return parser


def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if not args.eval:
        dataset_train = build_dataset(is_train="train", args=args)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        dataset_val = build_dataset(is_train="val", args=args)
        valid_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    dataset_test = build_dataset(is_train="test", args=args)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # initialise the model
    if "multimae" in args.output_dir:
        if "v1" not in args.init_weights:
            model = MultiMAEWrapper(
                input_size=(args.input_size, args.input_size),
                num_classes=args.nb_classes,
                classification=True,
            )
        else:
            model = MultiMAEWrapperv1(
                input_size=(args.input_size, args.input_size),
                num_classes=args.nb_classes,
                classification=True,
            )

        # load pre-trained weights if path provided
        if args.init_weights:
            model.load_weights(args.init_weights)
    else:
        model = vit_large_patch16(
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

        # load pre-trained weights and do same stuff as they do in RetFound
        if args.init_weights:
            checkpoint = torch.load(args.init_weights, map_location="cpu")

            print("Load pre-trained checkpoint from: %s" % args.init_weights)
            if "retfound" in args.output_dir:
                checkpoint_model = checkpoint["model"]
            else:
                checkpoint_model = checkpoint

            state_dict = model.state_dict()
            for k in ["head.weight", "head.bias"]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            # print(msg)

            if args.global_pool:
                assert set(msg.missing_keys) == {
                    "head.weight",
                    "head.bias",
                    "fc_norm.weight",
                    "fc_norm.bias",
                }
            else:
                assert set(msg.missing_keys) == {"head.weight", "head.bias"}

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)

    if args.linear_probing:
        # freeze encoder layers for linear probing
        if "multimae" in args.output_dir:
            for name, param in model.named_parameters():
                if "output_linear" not in name:
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

    model.to(device)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model,
        args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay,
        num_layers=len(model.model.encoder) if "multimae" in args.output_dir else None,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    if not args.eval:
        if args.smoothing > 0.0:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # Initialize early stopping object
        early_stopping = EarlyStopping(
            patience=args.early_stopping_epochs, delta=args.early_stopping_delta
        )

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_auc = 0.0
        train_stats_all, val_stats_all = [], []
        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_1_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                device,
                epoch,
                args=args,
            )

            train_stats_all.append(train_stats)

            val_stats = evaluate(
                model, valid_loader, epoch, device, args.nb_classes, mode="Valid"
            )
            val_stats_all.append(val_stats)

            # if the validation AUC has improved, save checkpoint
            if max_auc < val_stats[2]:
                max_auc = val_stats[2]

                if args.output_dir:
                    misc.save_model(args, epoch, model, optimizer)

            # Check if validation loss is decreasing
            early_stopping(val_stats[0])

            # Check if early stopping criterion is met
            if early_stopping.early_stop:
                print(f"Early stopping @ epoch {epoch}")
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    # Evaluate on the best checkpoint
    args.resume = f"{args.output_dir}/{args.task}/checkpoint-best-model-p-{args.train_ds_perc:.2f}.pth"
    misc.load_model(args=args, model=model, optimizer=optimizer)
    test_stats = evaluate(
        model, test_loader, "Best", device, args.nb_classes, mode="Test"
    )

    if not args.eval:
        return train_stats_all, val_stats_all, test_stats
    
    return test_stats


def append_results_to_csv(csv_path, res_stats):
    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open(csv_path, "a") as f_object:

        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)

        # Pass the rows as an argument into the writerow()
        for row in res_stats:
            writer_object.writerow(row)

        # Close the file object
        f_object.close()

init_weights_mapping = {
    "jx_vit_large_patch16_224_in21k.pth" : "ImageNet",
    "RETFound_oct_weights.pth" : "RetFound",
    "v1_bscan_512-224_checkpoint-1599.pth" : "MultiMAEv1",
    "bscan-slo_checkpoint-1599.pth" : "MultiMAEv2"
}

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.data_path = args.data_root + args.data_set

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{args.output_dir}/{args.task}").mkdir(parents=True, exist_ok=True)

    log_dir = "/msc/home/esueck47/scripts/foundoptima_clf_bsl/results"
    task, dataset = args.task.split("/")[0], args.task.split("/")[1]
    init_weights = init_weights_mapping[args.init_weights.split("/")[-1]] if args.init_weights else "Random"

    print(f"Dataset {dataset}; Init.weights: {init_weights}")
    print("-------------------------------------------")
    for p in [0.25, 0.5, 0.75, 1.0]:
        print(f"Training data % = {p:.3f}")
        args.train_ds_perc = p

        if not args.eval:
            train_stats, val_stats, test_stats = main(args)

            # columns: [Task, Dataset, Percentage, Epoch, Loss, BAcc, F1-score]
            append_results_to_csv(
                f"{log_dir}/train_eval_lab_eff_{task}.csv",
                [[init_weights, task, dataset, p] + r_i for r_i in train_stats],
            )

            # columns: [Task, Dataset, Percentage, Epoch, Loss, BAcc, AUROC, AP, F1-score, MCC]
            append_results_to_csv(
                f"{log_dir}/val_eval_lab_eff_{task}.csv",
                [[init_weights, task, dataset, p] + r_i for r_i in val_stats],
            )
        else:
            test_stats = main(args)

        # columns: [Task, Dataset, Percentage, BAcc, AUROC, AP, F1-score, MCC]
        append_results_to_csv(
            f"{log_dir}/test_eval_lab_eff_{task}.csv", [[init_weights, task, dataset, p] + test_stats[2:]]
        )

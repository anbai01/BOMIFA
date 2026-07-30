

import argparse
from training_pipeline import model_prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BOMIFA training")
    parser.add_argument("--data_folder", type=str, default="UCEC", help="Data folder name")
    parser.add_argument("--view_list", type=int, nargs="+", default=[1,2,3], help="List of omics views")
    parser.add_argument("--num_epoch_pretrain", type=int, default=600)
    parser.add_argument("--transformer_epochs", type=int, default=300)
    parser.add_argument("--lr_e_gcn", type=float, default=4e-5)
    parser.add_argument("--lr_e_cl_transformer", type=float, default=2e-6)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1600)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--cross_num_heads", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=6)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr_cross_attention", type=float, default=4e-6)
    parser.add_argument("--lr_c", type=float, default=8e-6)
    parser.add_argument("--all_lr", type=float, default=1e-6)
    parser.add_argument("--num_classes", type=int, default=2)

    args = parser.parse_args()

    model_prepare(
        args.data_folder,
        args.view_list,
        args.num_classes,
        args.lr_e_gcn,
        args.lr_e_cl_transformer,
        args.n_head,
        args.d_ff,
        args.num_layers,
        args.cross_num_heads,
        args.d_model,
        args.rank,
        args.lr_cross_attention,
        args.lr_c,
        args.all_lr,
        args.num_epoch_pretrain,
        args.transformer_epochs
    )
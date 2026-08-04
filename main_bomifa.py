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
    parser.add_argument("--output_folder", type=str, default="./preprocessed",
                        help="Folder containing preprocessed CSV files (X_train_*.csv, X_test_*.csv)")
    parser.add_argument("--top_num", type=int, default=100)

    args = parser.parse_args()

    model_prepare(
        data_folder=args.data_folder,
        view_list=args.view_list,
        num_class=args.num_classes,
        lr_e_gcn=args.lr_e_gcn,
        lr_e_cl_transformer=args.lr_e_cl_transformer,
        nhead=args.n_head,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        cross_num_heads=args.cross_num_heads,
        d_model=args.d_model,
        rank=args.rank,
        lr_cross_attention=args.lr_cross_attention,
        lr_c=args.lr_c,
        all_lr=args.all_lr,
        num_epoch_pretrain=args.num_epoch_pretrain,
        transformer_epochs=args.transformer_epochs,
        top_num=args.top_num,
        output_folder=args.output_folder
    )
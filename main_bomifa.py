from training_pipeline import model_prepare
if __name__ == "__main__":
    data_folder = 'UCEC'
    view_list = [1, 2, 3]
    num_epoch_pretrain = 600
    transformer_epochs = 300
    lr_e_gcn = 4e-5
    lr_e_cl_transformer = 1e-4
    n_head = 4
    d_ff = 1600
    num_layers = 10
    cross_num_heads =3
    d_model = 6
    rank =4
    lr_cross_attention = 4e-6
    lr_c = 8e-6
    all_lr = 1e-6
    num_classes = 2
    model_prepare(
        data_folder,view_list,num_classes,lr_e_gcn,lr_e_cl_transformer,n_head,d_ff,num_layers,
        cross_num_heads,d_model,rank,lr_cross_attention,lr_c,all_lr,num_epoch_pretrain,transformer_epochs)
import subprocess

# ***** MultiMAEv2 initialisation ***** #
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 2 --imgnet_scaler False --data_set OLIVES/  --task linprob/olives --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/bscan-slo_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v2'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 2 --imgnet_scaler False --data_set Harvard_Glaucoma/  --task linprob/harvard_glaucoma --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/bscan-slo_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v2'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 3 --imgnet_scaler False --data_set GAMMA/  --task linprob/gamma --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/bscan-slo_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v2'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 3 --imgnet_scaler False --batch_size 8 --data_set Duke_Srinivasan/  --task linprob/duke_srinivasan --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/bscan-slo_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v2'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 3 --imgnet_scaler False --data_set Noor_Eye_Hospital/  --task linprob/noor_eye_hospital --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/bscan-slo_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v2'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 5 --imgnet_scaler False --data_set 5C/  --task linprob/5c --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/bscan-slo_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v2'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 5 --imgnet_scaler False --data_set OCTID/  --task linprob/octid --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/bscan-slo_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v2'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 7 --imgnet_scaler False --data_set OCTDL/  --task linprob/octdl --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/bscan-slo_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v2'",
    shell=True,
)

# ***** MultiMAEv1 initialisation ***** #
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 2 --imgnet_scaler False --data_set OLIVES/  --task linprob/olives  --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1_bscan_512-224_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 2 --imgnet_scaler False --data_set Harvard_Glaucoma/  --task linprob/harvard_glaucoma  --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1_bscan_512-224_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 3 --imgnet_scaler False --data_set GAMMA/  --task linprob/gamma --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1_bscan_512-224_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 3 --imgnet_scaler False --batch_size 8 --data_set Duke_Srinivasan/  --task linprob/duke_srinivasan --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1_bscan_512-224_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 3 --imgnet_scaler False --data_set Noor_Eye_Hospital/  --task linprob/noor_eye_hospital --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1_bscan_512-224_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 5 --imgnet_scaler False --data_set 5C/  --task linprob/5c  --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1_bscan_512-224_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 5 --imgnet_scaler False --data_set OCTID/  --task linprob/octid  --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1_bscan_512-224_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 512 --nb_classes 7 --imgnet_scaler False --data_set OCTDL/  --task linprob/octdl  --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1_bscan_512-224_checkpoint-1599.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/multimae_exp/v1'",
    shell=True,
)

# ***** RetFound initialisation ***** #
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 2  --data_set OLIVES/  --task linprob/olives --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp/RETFound_oct_weights.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 2 --lr 1e-3 --data_set Harvard_Glaucoma/  --task linprob/harvard_glaucoma --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp/RETFound_oct_weights.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 3  --data_set GAMMA/  --task linprob/gamma --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp/RETFound_oct_weights.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 3 --batch_size 8 --data_set Duke_Srinivasan/  --task linprob/duke_srinivasan --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp/RETFound_oct_weights.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 3  --data_set Noor_Eye_Hospital/  --task linprob/noor_eye_hospital --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp/RETFound_oct_weights.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 5  --data_set 5C/  --task linprob/5c --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp/RETFound_oct_weights.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 5 --lr 1e-3 --data_set OCTID/  --task linprob/octid --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp/RETFound_oct_weights.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 7 --lr 1e-3 --data_set OCTDL/  --task linprob/octdl --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp/RETFound_oct_weights.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/retfound_exp'",
    shell=True,
)


# ***** ImageNet initialisation ***** #
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 2  --data_set OLIVES/  --task linprob/olives  --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp/jx_vit_large_patch16_224_in21k-606da67d.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 2 --lr 1e-3 --data_set Harvard_Glaucoma/  --task linprob/harvard_glaucoma --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp/jx_vit_large_patch16_224_in21k-606da67d.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 3  --data_set GAMMA/  --task linprob/gamma --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp/jx_vit_large_patch16_224_in21k-606da67d.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 3 --lr 1e-3 --batch_size 8 --data_set Duke_Srinivasan/  --task linprob/duke_srinivasan --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp/jx_vit_large_patch16_224_in21k-606da67d.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 3 --data_set Noor_Eye_Hospital/  --task linprob/noor_eye_hospital --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp/jx_vit_large_patch16_224_in21k-606da67d.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 5  --data_set 5C/  --task linprob/5c  --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp/jx_vit_large_patch16_224_in21k-606da67d.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 5  --data_set OCTID/  --task linprob/octid  --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp/jx_vit_large_patch16_224_in21k-606da67d.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp'",
    shell=True,
)
subprocess.run(
    "python clf_tasks_main.py --linear_probing True --lr 1e-3 --input_size 224 --nb_classes 7 --lr 1e-3 --data_set OCTDL/  --task linprob/octdl  --init_weights '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp/jx_vit_large_patch16_224_in21k-606da67d.pth' --output_dir '/msc/home/esueck47/scripts/foundoptima_clf_bsl/results/imgnet_exp'",
    shell=True,
)

from pathlib import Path
from os.path import lexists



base_path = '/mnt/Data/SSHFS/msc_server/MultiOptiMAE_downstream/classification_tasks/__results/21_minmax_baccloss_sf20-20/'
base_path = '/mnt/Data/SSHFS/msc_server/MultiOptiMAE_downstream/classification_tasks/__results/22_correct-checkpoint_minmax_baccloss_sf20-20/'
base_path = Path(base_path)

for rep in base_path.iterdir():
    # print(rep)
    for ft_type in rep.iterdir():
        if 'finetune' in ft_type.name:
            continue
        # print(ft_type)
        # for cross_dataset in ['UMN_Duke_Srinivasan_from_Noor']:
        for cross_dataset in ['UMN_Duke_Srinivasan_cross_test']:
            tgt_path = ft_type / cross_dataset
            tgt_path.mkdir(exist_ok=True)
            # shutil.rmtree(tgt_path)
            # continue
            noor_path = ft_type / 'Noor_Eye_Hospital_cross_train'
            for model in noor_path.iterdir():
                # if '21k' not in model.name:
                #     continue
                # print(model)
                tgt_path_model = tgt_path / model.name
                tgt_path_model.mkdir(exist_ok=True)
                checkpoint = model / 'checkpoint-best-model.pth'
                tgt_checkpoint = tgt_path_model / 'checkpoint-best-model.pth'
                if lexists(tgt_checkpoint):
                    continue
                    # tgt_checkpoint.unlink()
                print(tgt_path_model)
                tgt_checkpoint.symlink_to(
                    str(checkpoint).replace(
                        '/mnt/Data/SSHFS/msc_server/',
                        '/msc/home/jmoran82/'
                    )
                )

version=22_correct-checkpoint_minmax_baccloss_sf20-20
tab_path=/run/media/morano/SW1000/OPTIMA/Documents/My_Papers/MultiOptiMAE_Overleaf/tab
script=results_analysis/print_tex.py

python3 $script --version $version > $tab_path/22_sota_ft.tex
python3 $script --version $version --linear > $tab_path/22_sota_lin.tex
python3 $script --version $version --ablation > $tab_path/22_abl_ft.tex
python3 $script --version $version --ablation --linear > $tab_path/22_abl_lin.tex

#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=AdaBN_ImageNet
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-02:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/ENV/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/BMS .

echo "Copying the datasets"
date +"%T"
cp -r ~/scratch/CD-FSL_Datasets .

echo "creating data directories"
date +"%T"
cd BMS
cd data
unzip -q $SLURM_TMPDIR/CD-FSL_Datasets/miniImagenet.zip
unzip -q $SLURM_TMPDIR/CD-FSL_Datasets/ILSVRC_val.zip


echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"
echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR

cd BMS

# python AdaBN.py --dir ./logs/AdaBN_teacher/miniImageNet --base_dictionary logs/baseline_teacher/checkpoint_best.pkl --target_dataset ImageNet_test --target_subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --bsize 256 --epochs 10 --model resnet10
# python finetune.py --save_dir ./logs/AdaBN_teacher/miniImageNet --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/AdaBN_teacher/miniImageNet/checkpoint_best.pkl --freeze_backbone
# cp -r $SLURM_TMPDIR/BMS/logs/AdaBN_teacher/miniImageNet/ ~/scratch/BMS/logs/AdaBN_teacher/

python AdaBN.py --dir ./logs/AdaBN_na_teacher/miniImageNet --base_dictionary logs/baseline_na_teacher/checkpoint_best.pkl --target_dataset ImageNet_test --target_subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --bsize 256 --epochs 10 --model resnet10
python finetune.py --save_dir ./logs/AdaBN_na_teacher/miniImageNet --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/AdaBN_na_teacher/miniImageNet/checkpoint_best.pkl --freeze_backbone
cp -r $SLURM_TMPDIR/BMS/logs/AdaBN_na_teacher/miniImageNet/ ~/scratch/BMS/logs/AdaBN_na_teacher/

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
# echo "--------------------------------------<backup the result>-----------------------------------"
# date +"%T"
# cd $SLURM_TMPDIR
# cp -r $SLURM_TMPDIR/BMS/logs/AdaBN_teacher/miniImageNet/ ~/scratch/BMS/logs/AdaBN_teacher/
# cp -r $SLURM_TMPDIR/BMS/logs/AdaBN_na_teacher/miniImageNet/ ~/scratch/BMS/logs/AdaBN_na_teacher/

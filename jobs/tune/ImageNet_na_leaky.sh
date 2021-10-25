#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=ImageNet_na_leaky_tune
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-06:00
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

mkdir ChestX-Ray8 EuroSAT ISIC2018 plant-disease

cd EuroSAT
unzip ~/scratch/CD-FSL_Datasets/EuroSAT.zip
cd ..

cd ChestX-Ray8
unzip ~/scratch/CD-FSL_Datasets/ChestX-Ray8.zip
mkdir images
find . -type f -name '*.png' -print0 | xargs -0 mv -t images
cd ..

cd ISIC2018
unzip ~/scratch/CD-FSL_Datasets/ISIC2018.zip
unzip ~/scratch/CD-FSL_Datasets/ISIC2018_GroundTruth.zip
cd ..

cd plant-disease
unzip ~/scratch/CD-FSL_Datasets/plant-disease.zip

echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cd BMS

python ImageNet_finetune_leaky.py --save_dir ./logs/ImageNet_na_leaky/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/ImageNet_na_leaky/checkpoint_best.pkl --freeze_backbone &
python ImageNet_finetune_leaky.py --save_dir ./logs/ImageNet_na_leaky/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/ImageNet_na_leaky/checkpoint_best.pkl --freeze_backbone &
python ImageNet_finetune_leaky.py --save_dir ./logs/ImageNet_na_leaky/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/ImageNet_na_leaky/checkpoint_best.pkl --freeze_backbone &
wait
python ImageNet_finetune_leaky.py --save_dir ./logs/ImageNet_na_leaky/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/ImageNet_na_leaky/checkpoint_best.pkl --freeze_backbone



echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/BMS/logs/ImageNet_na_leaky/ ~/scratch/BMS/logs/
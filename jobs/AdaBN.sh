#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=AdaBN
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-08:00
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

target_testset="EuroSAT"
for target_testset in "ChestX" "ISIC" "EuroSAT" "CropDisease"
do
    python AdaBN.py --dir ./logs/AdaBN/$target_testset --base_dictionary logs/AdaBN/teacher_miniImageNet/399.tar --target_dataset $target_testset --target_subset_split datasets/split_seed_1/$target_testset\_unlabeled_20.csv --bsize 256 --epochs 10 --model resnet10 &
done
wait

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/BMS/logs/AdaBN/ ~/scratch/BMS/logs/

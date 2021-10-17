#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=ImageNet_val
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-04:00
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
cp ~/scratch/imagenet_object_localization_patched2019.tar.gz .

echo "creating data directories"
date +"%T"
cd BMS
cd data
tar -xzf $SLURM_TMPDIR/imagenet_object_localization_patched2019.tar.gz
unzip -q $SLURM_TMPDIR/CD-FSL_Datasets/ILSVRC_val.zip

echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cd BMS

echo "---------------------------------------ImageNet------------------------------------"
python ImageNet.py --resume ./logs/ImageNet/checkpoint_best.pkl --evaluate --arch resnet18 --data ./data/ILSVRC/Data/CLS-LOC --gpu 0
echo "---------------------------------------------------------------------------"

echo "---------------------------------------ImageNet_na------------------------------------"
python ImageNet_na.py --resume ./logs/ImageNet_na/checkpoint_best.pkl --evaluate --arch resnet18 --data ./data/ILSVRC/Data/CLS-LOC --gpu 0
echo "---------------------------------------------------------------------------"

echo "---------------------------------------ImageNet_nb------------------------------------"
python ImageNet_nb.py --resume ./logs/ImageNet_nb/checkpoint_best.pkl --evaluate --arch resnet18 --data ./data/ILSVRC/Data/CLS-LOC --gpu 0
echo "---------------------------------------------------------------------------"

echo "---------------------------------------ImageNet_nw------------------------------------"
python ImageNet_nw.py --resume ./logs/ImageNet_nw/checkpoint_best.pkl --evaluate --arch resnet18 --data ./data/ILSVRC/Data/CLS-LOC --gpu 0
echo "---------------------------------------------------------------------------"

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"

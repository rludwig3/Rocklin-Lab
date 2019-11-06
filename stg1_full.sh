#!/bin/bash
#SBATCH -A p30802               
#SBATCH -p normal               
#SBATCH -t 10:00:00            
#SBATCH -N 1                    
#SBATCH --mem=24G               
#SBATCH --ntasks-per-node=6     
#SBATCH --mail-user=robert.ludwig@northwestern.edu 
#SBATCH --mail-type=END  
#SBATCH --array=0-22
#SBATCH --output=/projects/p30802/wes/hdx_analysis/stg1/"%a_%A.o"
#SBATCH --error=/projects/p30802/wes/hdx_analysis/stg1/"%a_%A.e"
#SBATCH --job-name=HDX_scan_extraction_full

# unload any modules that carried over from your command line session
module purge

# load modules you need to use
module load python/anaconda3.6
python stg1.py ${SLURM_ARRAY_TASK_ID}

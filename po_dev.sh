#!/bin/bash
#SBATCH -A p30802               # Allocation
#SBATCH -p normal                # Queue
#SBATCH -t 48:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=64G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=4     # Number of Cores (Processors)
#SBATCH --mail-user=robert.ludwig@northwestern.edu  # Designate email address for job communications
#SBATCH --mail-type=END    # Events options are job BEGIN, END, NONE, FAIL, REQUEUE
#SBATCH --array=1-890 
#SBATCH --output=/projects/p30802/wes/hdx_analysis/stg2/po_testing/%A_%a.o    # Path for output must already exist
#SBATCH --error=/projects/p30802/wes/hdx_analysis/stg2/po_testing/%A_%a.e    # Path for errors must already exist
#SBATCH --job-name=LCMSTA_2     # Name of job

# unload any modules that carried over from your command line session
module purge

# load modules you need to use
#pass name of protein to python argv by putting it as argument after .sh file. EX: sbatch stg2_single_name_short.sh EEHEE_rd3_0189.pdb
module load python/anaconda3.6
source activate dask_lab
python stg2_dev_11_4.py $(sed -n ${SLURM_ARRAY_TASK_ID}p name_list.csv | sed 's/,//g')

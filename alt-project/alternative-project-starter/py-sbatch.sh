#!/bin/bash
# Run a Python script on a lambda compute node as a slurm batch job.
#
# Usage, from the project folder on lambda:
#   ./py-sbatch.sh part2_planning.py --id 123456789
#
# The output goes to slurm-<job id>.out in the current folder.
# Never run long computations on the lambda gateway itself.

NUM_CORES=2
JOB_NAME="parlai-alt"
MAIL_USER=""                       # optional, your email for a notice when the job ends
CONDA_HOME="$HOME/miniconda3"      # change if conda is installed elsewhere
CONDA_ENV="parlai-alt"

MAIL_ARGS=""
if [ -n "$MAIL_USER" ]; then
  MAIL_ARGS="--mail-type=END,FAIL --mail-user=$MAIL_USER"
fi

sbatch -c "$NUM_CORES" -J "$JOB_NAME" -o "slurm-%j.out" $MAIL_ARGS <<EOF
#!/bin/bash
echo "*** job \$SLURM_JOB_ID on \$(hostname) started \$(date)"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export MPLBACKEND=Agg
python $@
echo "*** job \$SLURM_JOB_ID finished \$(date)"
EOF

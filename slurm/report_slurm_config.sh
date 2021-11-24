#!/bin/bash

echo ""
echo "-------- Reporting SLURM configuration ---------------------------------"
date
echo ""
echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) begins on $SLURM_NODENAME, submitted from $SLURM_SUBMIT_HOST ($SLURM_CLUSTER_NAME)"
echo ""
echo "SLURM_CLUSTER_NAME      = $SLURM_CLUSTER_NAME"        # Name of the cluster on which the job is executing.
echo "SLURM_NODENAME          = $SLURM_NODENAME"
echo "SLURM_JOB_QOS           = $SLURM_JOB_QOS"             # Quality Of Service (QOS) of the job allocation.
echo "SLURM_JOB_ID            = $SLURM_JOB_ID"              # The ID of the job allocation.
echo "SLURM_RESTART_COUNT     = $SLURM_RESTART_COUNT"       # The number of times the job has been restarted.
if [ "$SLURM_ARRAY_TASK_COUNT" != "" ] && [ "$SLURM_ARRAY_TASK_COUNT" -gt 1 ]; then
    echo ""
    echo "SLURM_ARRAY_JOB_ID      = $SLURM_ARRAY_JOB_ID"        # Job array's master job ID number.
    echo "SLURM_ARRAY_TASK_COUNT  = $SLURM_ARRAY_TASK_COUNT"    # Total number of tasks in a job array.
    echo "SLURM_ARRAY_TASK_ID     = $SLURM_ARRAY_TASK_ID"       # Job array ID (index) number.
    echo "SLURM_ARRAY_TASK_MAX    = $SLURM_ARRAY_TASK_MAX"      # Job array's maximum ID (index) number.
    echo "SLURM_ARRAY_TASK_STEP   = $SLURM_ARRAY_TASK_STEP"     # Job array's index step size.
fi;
echo ""
echo "SLURM_JOB_NUM_NODES     = $SLURM_JOB_NUM_NODES"       # Total number of nodes in the job's resource allocation.
echo "SLURM_JOB_NODELIST      = $SLURM_JOB_NODELIST"        # List of nodes allocated to the job.
echo "SLURM_TASKS_PER_NODE    = $SLURM_TASKS_PER_NODE"      # Number of tasks to be initiated on each node.
echo "SLURM_NTASKS            = $SLURM_NTASKS"              # Number of tasks to spawn.
echo "SLURM_PROCID            = $SLURM_PROCID"              # The MPI rank (or relative process ID) of the current process
echo ""
echo "SBATCH_GRES             = $SBATCH_GRES"               # Same as --gres
echo "SLURM_GRES              = $SLURM_GRES"                #
echo "SBATCH_GPU_BIND         = $SBATCH_GPU_BIND"           # Same as --gpu-bind
echo "SLURM_GPU_BIND          = $SLURM_GPU_BIND"            # Requested binding of tasks to GPU. Only set if the --gpu-bind option is specified.
echo "SBATCH_GPU_FREQ         = $SBATCH_GPU_FREQ"           # Number of GPUs allocated to the batch step.
echo "SLURM_GPU_FREQ          = $SLURM_GPU_FREQ"            # Requested GPU frequency. Only set if the --gpu-freq option is specified.
echo "SBATCH_GPUS             = $SBATCH_GPUS"               # Same as -G, --gpus
echo "SLURM_GPUS              = $SLURM_GPUS"                # Number of GPUs requested. Only set if the -G, --gpus option is specified.
echo "SBATCH_GRES_FLAGS       = $SBATCH_GRES_FLAGS"         # Same as --gres-flags
echo "SLURM_GPUS_ON_NODE      = $SLURM_GPUS_ON_NODE"        # Number of GPUs allocated to the batch step.
echo "SLURM_GPUS_PER_NODE     = $SLURM_GPUS_PER_NODE"       # Requested GPU count per allocated node. Only set if the --gpus-per-node option is specified
echo "SLURM_GPUS_PER_SOCKET   = $SLURM_GPUS_PER_SOCKET"     # Requested GPU count per allocated socket. Only set if the --gpus-per-socket option is specified.
echo "SLURM_GPUS_PER_TASK     = $SLURM_GPUS_PER_TASK"       # Requested GPU count per allocated task. Only set if the --gpus-per-task option is specified.
echo ""
echo "GPUS_PER_NODE           = $GPUS_PER_NODE"             # Manually set number of GPUs.
echo "SLURM_CPUS_ON_NODE      = $SLURM_CPUS_ON_NODE"        # Number of CPUs allocated to the batch step.
echo "SLURM_JOB_CPUS_PER_NODE = $SLURM_JOB_CPUS_PER_NODE"   # Count of CPUs available to the job on the nodes in the allocation.
echo "SLURM_CPUS_PER_TASK     = $SLURM_CPUS_PER_TASK"       # Number of cpus requested per task. Only set if the --cpus-per-task option is specified.
echo "SLURM_MEM_PER_CPU       = $SLURM_MEM_PER_CPU"         # Same as --mem-per-cpu
echo "SLURM_MEM_PER_GPU       = $SLURM_MEM_PER_GPU"         # Same as --mem-per-gpu
echo "SLURM_MEM_PER_NODE      = $SLURM_MEM_PER_NODE"        # Same as --mem
echo ""
echo "------------------------------------"
echo ""
if [[ "$SLURM_TMPDIR" != "" ]];
then
    echo "SLURM_TMPDIR = $SLURM_TMPDIR"
    echo ""
    echo "Contents of $SLURM_TMPDIR"
    ls -lh "$SLURM_TMPDIR"
    echo ""
fi;
echo "df -h:"
df -h
echo ""
if [[ "$start_time" != "" ]];
then
    echo "------------------------------------"
    elapsed=$(( SECONDS - start_time ))
    eval "echo Running total elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
fi
echo "------------------------------------------------------------------------"

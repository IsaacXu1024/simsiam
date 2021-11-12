#!/bin/bash

echo ""
echo "-------- Reporting environment configuration ---------------------------"
date
echo ""
echo "pwd:"
pwd
echo ""
echo "which python:"
which python
echo ""
echo "python version:"
python --version
echo ""
echo "which conda:"
which conda
echo ""
echo "conda info:"
conda info
echo ""
echo "conda env export:"
conda env export
echo ""
echo "which pip:"
which pip
echo ""
echo "pip freeze:"
echo ""
pip freeze
echo ""
echo "which nvcc:"
which nvcc
echo ""
echo "nvcc version:"
nvcc --version
echo ""
echo "nvidia-smi:"
nvidia-smi
echo ""
echo "torch info:"
python -c "import torch; print('pytorch={}, cuda={}, gpus={}'.format(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()))"
python -c "import torch; print(str(torch.ones(1, device=torch.device('cuda')))); print('able to use cuda')"
echo ""
if [[ "$start_time" != "" ]];
then
    echo "------------------------------------"
    elapsed=$(( SECONDS - start_time ))
    eval "echo Running total elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
fi
echo "------------------------------------------------------------------------"

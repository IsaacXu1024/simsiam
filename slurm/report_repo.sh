#!/bin/bash

echo ""
echo "-------- Reporting git repo configuration ------------------------------"
date
echo ""
echo "pwd:"
pwd
echo ""
echo "commit ref:"
git rev-parse HEAD
echo ""
git status
echo ""
if [[ "$start_time" != "" ]];
then
    echo "------------------------------------"
    elapsed=$(( SECONDS - start_time ))
    eval "echo Running total elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
fi
echo "------------------------------------------------------------------------"

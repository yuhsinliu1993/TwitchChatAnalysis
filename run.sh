#!/bin/bash

K=$1
alpha=`echo "scale=3;50/$K"|bc`
beta=0.005
niter=5
save_step=501

streamer=$2
output_dir=./streamer/${streamer}/output/
model_dir=${output_dir}model/
dwid_pt=${output_dir}doc_wids.txt
voca_pt=${output_dir}vocabulary.txt

echo "=============== Topic Learning ============="
W=`wc -l < $voca_pt` # vocabulary size
echo "./src/btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir"
./src/btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir


# infer p(z|d) for each doc
echo "================ Infer P(z|d)==============="
echo "./BTM/src/btm inf sum_b $K $dwid_pt $model_dir"
./BTM/src/btm inf sum_b $K $dwid_pt $model_dir



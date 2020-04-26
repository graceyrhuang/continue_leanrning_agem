GPUID=-1
OUTDIR=outputs/permuted_MNIST_incremental_class
REPEAT=10
python ../iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --n_permutation 10 --force_out_dim 100 --schedule 10 --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name AGEM_4000   --lr 0.1 --reg_coef 0.5          | tee ${OUTDIR}/AGEM_4000.log
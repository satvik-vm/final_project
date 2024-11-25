python3 train.py \
    --dataset miniImagenet \
    --method manifold_mixup \
    --model WideResNet28_10 \
    --batch_size 250 \
    --test_batch_size 250 \
    --stop_epoch 400 \
    --save_freq 10 \
    --num_classes 10 \
    --print_batch_freq 10 \
	--alpha 5 \
	--bottleneck 32\
	--without_mixup
	# 2>&1 | tee checkpoints/temp/console_output.txt

# python experiments/mnist/baseline.py --device=cuda:0 --seed=0
# python experiments/mnist/random_route.py --device=cuda:0 --seed=0
# python experiments/mnist/receiver_first.py --device=cuda:0 --outlier_method=ground_truth --seed=0

# run for 5 seeds
# for seed in 0 1 2 3 4
for seed in 1 2 3 4
do
    python experiments/fashion_mnist/baseline.py --device=cuda:0 --seed=$seed
    python experiments/fashion_mnist/receiver_first.py --device=cuda:0 --outlier_method=pythresh_cac --query_strategy=combine_wrong_least_confidence --seed=$seed
    # python experiments/fashion_mnist/random_route.py --device=cuda:0 --seed=$seed
    # python experiments/mnist/receiver_first.py --device=cuda:0 --outlier_method=pythresh_cac --query_strategy=combine_wrong_least_confidence --seed=$seed --enable_funky_threshold
    # not good outlier method
    # python experiments/mnist/receiver_first.py --device=cuda:0 --outlier_method=cac_contrastive --query_strategy=combine_wrong_least_confidence --seed=$seed
done
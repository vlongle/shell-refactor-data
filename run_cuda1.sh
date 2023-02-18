# python experiments/mnist/receiver_first.py --device=cuda:1 --outlier_method=no_outlier_detection --seed=0
# python experiments/mnist/receiver_first.py --device=cuda:1 --outlier_method=cac_contrastive --seed=0




# python experiments/mnist/receiver_first.py --device=cuda:0 --outlier_method=cac_contrastive --seed=0
# python experiments/mnist/receiver_first.py --device=cuda:1 --outlier_method=pythresh_cac --seed=0
# python experiments/mnist/receiver_first.py --device=cuda:0 --outlier_method=pythresh_detector --seed=0



# python experiments/mnist/receiver_first.py --device=cuda:0 --outlier_method=pythresh_cac --query_strategy=combine_wrong_least_confidence --seed=0
# for seed in 0 1 2 3 4
for seed in 1 2 3 4
do
    python experiments/fashion_mnist/random_route.py --device=cuda:1 --seed=$seed
    python experiments/fashion_mnist/receiver_first.py --device=cuda:1 --outlier_method=ground_truth --query_strategy=combine_wrong_least_confidence --seed=$seed
    # python experiments/mnist/receiver_first.py --device=cuda:1 --outlier_method=pythresh_cac --query_strategy=combine_wrong_least_confidence --seed=$seed
    # python experiments/mnist/random_route.py --device=cuda:1 --seed=$seed
done
for seed in 0 1 2 3 4
do
    python experiments/mnist/receiver_first_more_data.py --device=cuda:1 --outlier_method=pythresh_cac--query_strategy=combine_wrong_least_confidence --seed=$seed
    python experiments/mnist/receiver_first_more_data.py --device=cuda:0 --outlier_method=pythresh_cac--query_strategy=combine_wrong_least_confidence --seed=$seed --enable_funky_threshold
done
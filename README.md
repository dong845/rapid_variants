# rapid_variants

## rapid-origin:
original code of rapid

## rapid-structure:
change model struture, run 'python main.py --env MiniGrid-KeyCorridorS3R2-v0 --sl_until 1200000 --model_type=mlp', model_type can be "mlp", "cnn1d", "cnn2d"

## rapid-sample:
change sample method, run 'python main.py --env MiniGrid-KeyCorridorS3R2-v0 --sl_until 1200000 --choice_type=original' choice_type can be "original", "softmax" and "epsilon-greedy"

## rapid-weight:
give larger weight to beneficial states (key, box and door), run 'python main.py --env MiniGrid-KeyCorridorS3R2-v0 --sl_until 1200000 --local_score_type=original' local_score_type can be "original" and "new"

## rapid-anneal-ratio:
anneal weight of local score and global store with different start point based on setting ratio, run 'python main.py --env MiniGrid-KeyCorridorS3R2-v0 --sl_until 1200000 --ratio=0.5' ratio can be any float number less than 1, we pick 0.25, 0.5 and 0.75

## rapid-anneal:
anneal weight of local score and global store when extrinsic reward is not 0, run just 'python main.py --env MiniGrid-KeyCorridorS3R2-v0 --sl_until 1200000'

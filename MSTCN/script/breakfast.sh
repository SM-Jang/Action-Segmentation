dataset='breakfast'
op='residual'
gpu=2


method='triplet_9'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

method='triplet_10'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

method='triplet_11'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

method='triplet_12'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

method='triplet_13'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

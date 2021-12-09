dataset='gtea'
op='residual'
gpu=2

method='triplet_4'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

method='triplet_5'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

method='triplet_6'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

method='triplet_7'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

method='triplet_8'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

method='triplet_9'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method

method='triplet_10'
python main.py --op=$op --action='train' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='test' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
python main.py --op=$op --action='eval' --dataset=$dataset --split='4' --gpu=$gpu --sample_rate=1 --method=$method
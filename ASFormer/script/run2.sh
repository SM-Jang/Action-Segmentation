dataset=50salads
op='residual'

gpu=2


method='triplet_7'
python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python eval.py --dataset=$dataset --split=4 --method=$method

method='triplet_8'
python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python eval.py --dataset=$dataset --split=4 --method=$method

method='triplet_9'
python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python eval.py --dataset=$dataset --split=4 --method=$method


method='triplet_11'
python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python eval.py --dataset=$dataset --split=4 --method=$method


method='triplet_12'
python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python eval.py --dataset=$dataset --split=4 --method=$method

method='triplet_13'
python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python eval.py --dataset=$dataset --split=4 --method=$method

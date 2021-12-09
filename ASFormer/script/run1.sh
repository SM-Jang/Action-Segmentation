dataset=gtea
op='residual'
gpu=3



# method='triplet_5'
# python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
# python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
# python eval.py --dataset=$dataset --split=4 --method=$method

# method='triplet_6'
# python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
# python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
# python eval.py --dataset=$dataset --split=4 --method=$method

method='triplet_7'
python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python eval.py --dataset=$dataset --split=4 --method=$method

method='triplet_8'
python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python eval.py --dataset=$dataset --split=4 --method=$method


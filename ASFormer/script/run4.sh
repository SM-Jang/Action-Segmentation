dataset=breakfast
op='residual'
gpu=1



method='triplet_13'
python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python eval.py --dataset=$dataset --split=4 --method=$method

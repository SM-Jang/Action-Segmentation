dataset=breakfast
op='residual'
gpu=3




method='triplet_10'
python main.py --action=train --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python main.py --action=predict --dataset=$dataset --split=4 --gpu=$gpu --op=$op --method=$method
python eval.py --dataset=$dataset --split=4 --method=$method

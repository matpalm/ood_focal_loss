# out of distribution detection using focal loss

see [blog post](http://matpalm.com/blog/ood_using_focal_loss/)

## prepare data splits

```
python3 prepare_splits.py
```

## model_1 experiments; baseline

```
python3 train_basic_model.py --loss-fn cross_entropy --model-dir m1
python3 calculate_entropies.py --model-dir m1
```

## model_2 experiments; finetune classifier; aka logisitic regression calibration

```
# model_1 experiments
python3 train_model_2.py --input-model-dir m1 --output-model-dir m2
python3 calculate_metrics.py --model m1/weights.npz
python3 calculate_metrics.py --model m2/weights.npz
python3 calculate_entropies.py --model-dir m2
```

## model_3 experiments; fit scalar temperture

```
# model_1 experiments
python3 train_model_3.py --input-model-dir m1 --output-model-dir m3
python3 calculate_entropies.py --model-dir m3
python3 calculate_entropies.py --model-dir m3 --include-temp-layer
python3 calculate_metrics.py --model m3/weights.npz
python3 calculate_metrics.py --model m3/weights.npz --include-temp-layer
```

## model_4 experiments; use focal loss

```
python3 train_basic_model.py --loss-fn focal_loss --gamma 1.0 --model-dir m4_1
python3 calculate_entropies.py --model-dir m4_1
python3 train_basic_model.py --loss-fn focal_loss --gamma 2.0 --model-dir m4_2
python3 calculate_entropies.py --model-dir m4_2
python3 train_basic_model.py --loss-fn focal_loss --gamma 3.0 --model-dir m4_3
python3 calculate_entropies.py --model-dir m4_3
```
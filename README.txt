# Discovery of Deep Options

Discovery of Deep Optionsのモデルの実装とGymのTaxiv3に対しての動作確認。

論文： [Multi-Level Discovery of Deep Options](https://arxiv.org/pdf/1703.08294.pdf)


## 学習

`train.py`でモデルを学習する：

```
$> python3 train.py
```

モデルのCheckpointが`./saves/<datetime>/`の中に保存される。
モデルのTensorboardのlogsが`./logs/<datetime>/`の中に保存される。

## 動作確認

`eval.py <checkpoint>`でモデルの動作を確認できる。

```
$> python3 eval.py trained_checkpoints/option4.chkpt
```

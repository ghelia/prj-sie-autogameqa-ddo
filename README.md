# Discovery of Deep Options

Discovery of Deep Optionsのモデルの実装とGymのTaxiv3に対しての動作確認。

論文： [Multi-Level Discovery of Deep Options](https://arxiv.org/pdf/1703.08294.pdf)


## 開発環境

`Python 3.9.10`でテストされた

モジュールをインストールするために以下のコマンドを実行する。

```
python setup.py install
```


## 学習

`train.py`でモデルを学習する：

```
$> python train.py
```

モデルのCheckpointが`./saves/<datetime>/`の中に保存される。
モデルのTensorboardのlogsが`./logs/<datetime>/`の中に保存される。`tensorboad --logdir ./logs`で見られる。


## 動作確認

`eval.py <checkpoint>`でモデルの動作を確認できる。

```
$> python eval.py trained_checkpoints/option4.chkpt
```

![alt text](https://github.com/ghelia/prj-sie-autogameqa-ddo/blob/master/ddo-taxiv3.gif)


## ユニッテスト


ユニッテストをチェックするために以下のコマンドを実行する。

```
$> pytest tests
```

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fastai/fastbook/master)  
[English](./README.md) / [Spanish](./README_es.md) / [Korean](./README_ko.md) / [Chinese](./README_zh.md) / [Bengali](./README_bn.md) / [Indonesian](./README_id.md) / [Italian](./README_it.md) / [Portuguese](./README_pt.md) / [Vietnamese](./README_vn.md) / [Japanese](./README_ja.md)

# The fastai book

このレポジトリにあるノートブックは、ディープラーニング、[fastai](https://docs.fast.ai/)、 [PyTorch](https://pytorch.org/)の入門を扱っています。fastaiはディープラーニングのための階層型のAPIを提供します。更に詳しく学びたい方は[the fastai paper](https://www.mdpi.com/2078-2489/11/2/108)をお読みください。このレポジトリの全てはJeremy HowardとSylvain Guggerが2020年以降の著作権を保持します。一部のセクションは[オンラインで読む](https://fastai.github.io/fastbook2e/)ことができます。

このレポジトリにあるノートブックは[a MOOC](https://course.fast.ai)で使われており、また[この本](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527)のベースとなっています。このレポジトリにあるようなGPLの制限はありません。

このレポジトリにあるノートブック内のコードやPythonのファイル`.py`はGPL v3ライセンスで公開されています。詳細は[LICENSE ファイル](./LICENSE)を参照してください。それ以外のもの（すべてのマークダウンセルやその他文章を含む）は私的利用の目的のみにおいて、再配布やフォーマット・媒体の変更、ノートブックのコピーを作成したり、レポジトリをフォークすることが可能です。営利目的、放映目的での利用は認められません。多くの方がディープラーニングを学ぶのを手助けするために自由に利用できるようにしていますので、著作権とこれらの制限を尊重してください。

もし、これらの資料のコピーをどこかでホストしている人を見かけたら、その行為が許されないものであり、法的措置につながる可能性があることを教えてあげてください。また、この著作権を無視する人がいるようであれば、私たちは追加の資料を公開できなくなるため、コミュニティにも損害を与えることになります。

## Colab

このレポジトリをクローンして自分のマシンで開く代わりに、[Google Colab](https://research.google.com/colaboratory/)を使ってノートブックを読んだり、作業することができます。この方法はPythonの開発環境をセットアップする必要がなく、ブラウザ上ですぐに作業を始められるので、勉強を始められたばかりの方には特におすすめです。

この本の任意の章を次のリンクから開くことができます:: [Introduction to Jupyter](https://colab.research.google.com/github/fastai/fastbook/blob/master/app_jupyter.ipynb) | [Chapter 1, Intro](https://colab.research.google.com/github/fastai/fastbook/blob/master/01_intro.ipynb) | [Chapter 2, Production](https://colab.research.google.com/github/fastai/fastbook/blob/master/02_production.ipynb) | [Chapter 3, Ethics](https://colab.research.google.com/github/fastai/fastbook/blob/master/03_ethics.ipynb) | [Chapter 4, MNIST Basics](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb) | [Chapter 5, Pet Breeds](https://colab.research.google.com/github/fastai/fastbook/blob/master/05_pet_breeds.ipynb) | [Chapter 6, Multi-Category](https://colab.research.google.com/github/fastai/fastbook/blob/master/06_multicat.ipynb) | [Chapter 7, Sizing and TTA](https://colab.research.google.com/github/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb) | [Chapter 8, Collab](https://colab.research.google.com/github/fastai/fastbook/blob/master/08_collab.ipynb) | [Chapter 9, Tabular](https://colab.research.google.com/github/fastai/fastbook/blob/master/09_tabular.ipynb) | [Chapter 10, NLP](https://colab.research.google.com/github/fastai/fastbook/blob/master/10_nlp.ipynb) | [Chapter 11, Mid-Level API](https://colab.research.google.com/github/fastai/fastbook/blob/master/11_midlevel_data.ipynb) | [Chapter 12, NLP Deep-Dive](https://colab.research.google.com/github/fastai/fastbook/blob/master/12_nlp_dive.ipynb) | [Chapter 13, Convolutions](https://colab.research.google.com/github/fastai/fastbook/blob/master/13_convolutions.ipynb) | [Chapter 14, Resnet](https://colab.research.google.com/github/fastai/fastbook/blob/master/14_resnet.ipynb) | [Chapter 15, Arch Details](https://colab.research.google.com/github/fastai/fastbook/blob/master/15_arch_details.ipynb) | [Chapter 16, Optimizers and Callbacks](https://colab.research.google.com/github/fastai/fastbook/blob/master/16_accel_sgd.ipynb) | [Chapter 17, Foundations](https://colab.research.google.com/github/fastai/fastbook/blob/master/17_foundations.ipynb) | [Chapter 18, GradCAM](https://colab.research.google.com/github/fastai/fastbook/blob/master/18_CAM.ipynb) | [Chapter 19, Learner](https://colab.research.google.com/github/fastai/fastbook/blob/master/19_learner.ipynb) | [Chapter 20, conclusion](https://colab.research.google.com/github/fastai/fastbook/blob/master/20_conclusion.ipynb)


## コントリビューション

このレポジトリに対してプルリクエストをした場合、その著作権はすべてJeremy HowardとSylvain Guggerに譲渡されます。（また、テキストやスペルの修正などの小さな変更を加える場合は、修正をするファイル名と修正事項に関する簡潔な説明文を含めてください。でないとレビュアーにはどこが修正されているか判断するのが困難です。よろしくおねがいします。）

## 引用

この本を引用する場合は下記の内容をお使いください:

```
@book{howard2020deep,
title={Deep Learning for Coders with Fastai and Pytorch: AI Applications Without a PhD},
author={Howard, J. and Gugger, S.},
isbn={9781492045526},
url={https://books.google.no/books?id=xd6LxgEACAAJ},
year={2020},
publisher={O'Reilly Media, Incorporated}
}
```


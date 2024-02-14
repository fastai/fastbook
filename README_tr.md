[English](./README.md) / [Spanish](./README_es.md) / [Korean](./README_ko.md) / [Chinese](./README_zh.md) / [Bengali](./README_bn.md) / [Indonesian](./README_id.md) / [Italian](./README_it.md) / [Portuguese](./README_pt.md) / [Vietnamese](./README_vn.md) / [Japanese](./README_ja.md) / [Turkish](./README_tr.md)
# Fastai kitabı

Bu Jupyter Notebook'ları, Derin Öğrenme, [fastai](https://docs.fast.ai/) ve [PyTorch](https://pytorch.org/) hakkında eğitim içermektedir. Fastai, Derin Öğrenme tekniklerine özel katmanlı API içermektedir; daha fazla bilgi için [bkz.](https://www.mdpi.com/2078-2489/11/2/108) Bu Repodaki her şeyin telif hakkı, 2020'den itiabaren, Jeremy Howard ve Sylvain Gugger'a aittir. Seçili okunabilen bölümlere [buradan ulaşabilirsiniz](https://fastai.github.io/fastbook2e/).

Repoda bulunan Jupyter Notebook'ları, [bu çevrimiçi kurs](https://course.fast.ai) için hazırlanmıştır. Ayrıca, şuan satılık olan, [bu kitabın](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527) temellerini oluşturmaktadır.
Kitap, aynı GPL kısıtlamalarına tabii değildir.

Jupyter Notebook'ları ve python `.py` dosyalarının tamamı GPL v3 ile lisanslanmıştı; daha fazla detay için LICENSE'a bakınız. Geri kalanı (notebooklarda bulunan markdown hücreleri dahil olmak üzere), herhangi bir yeniden dağıtım veya format değişmi için, Repo'yu çatallamak ve özel kullanımınız hariç, lisanslanmamıştır. Ticari veya yayın amaçlı kullanıma izin verilmez. Deri Öğrenme konularını öğrenebilmeniz amacıyla bu içeriği ücretsiz sunuyoruz. Dolayısıyla, lütfen bahsi geçen kısıtlamalara ve telif hakkına saygı gösteriniz. 

Bu içeriği başka yerlerde yayınlanan birisini görürseniz lütfen yaptıklarının izinli olmadığını ve onlar hakkında soruşturma başlatılabileceğini söyleyiniz. Dahası, topluluğa da zarar vermiş olacaklardır çünkü eğer insanlar telif hakkımızı göz ardı ederse bu şekilde içerik yayınlamamız pek olası değildir.

## Colab

Repo'yu klonlayıp bilgisayarınızda çalıştırmak yerine, [Google Colab](https://research.google.com/colaboratory/) kullanarak Notebook'ları çalışabilirsiniz. Bu, yeni başlayanlar için önerilen yaklaşımdır; doğrudan web tarayıcınızda çalışabileceğiniz için kendi makinenizde bir Python geliştirme ortamı kurmanıza da gerek yoktur.

Kitabın herhangi bir bölümüne Google Colab üzerine bu linklerden ulaşabilirsiniz: [Jupyter'e Giriş](https://colab.research.google.com/github/fastai/fastbook/blob/master/app_jupyter.ipynb) | [Bölüm 1, Giriş](https://colab.research.google.com/github/fastai/fastbook/blob/master/01_intro.ipynb) | [Bölüm 2, Production](https://colab.research.google.com/github/fastai/fastbook/blob/master/02_production.ipynb) | [Bölüm 3, Etikler](https://colab.research.google.com/github/fastai/fastbook/blob/master/03_ethics.ipynb) | [Bölüm 4, MNIST temelleri](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb) | [Bölüm 5, Pet Breeds](https://colab.research.google.com/github/fastai/fastbook/blob/master/05_pet_breeds.ipynb) | [Bölüm 6, Multi-Category](https://colab.research.google.com/github/fastai/fastbook/blob/master/06_multicat.ipynb) | [Bölüm 7, Sizing and TTA](https://colab.research.google.com/github/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb) | [Bölüm 8, Collab](https://colab.research.google.com/github/fastai/fastbook/blob/master/08_collab.ipynb) | [Bölüm 9, Tabular](https://colab.research.google.com/github/fastai/fastbook/blob/master/09_tabular.ipynb) | [Bölüm 10, NLP](https://colab.research.google.com/github/fastai/fastbook/blob/master/10_nlp.ipynb) | [Bölüm 11, Mid-Level API](https://colab.research.google.com/github/fastai/fastbook/blob/master/11_midlevel_data.ipynb) | [Bölüm 12, NLP Deep-Dive](https://colab.research.google.com/github/fastai/fastbook/blob/master/12_nlp_dive.ipynb) | [Bölüm 13, Convolutions](https://colab.research.google.com/github/fastai/fastbook/blob/master/13_convolutions.ipynb) | [Bölüm 14, Resnet](https://colab.research.google.com/github/fastai/fastbook/blob/master/14_resnet.ipynb) | [Bölüm 15, Arch Details](https://colab.research.google.com/github/fastai/fastbook/blob/master/15_arch_details.ipynb) | [Bölüm 16, Optimizers and Callbacks](https://colab.research.google.com/github/fastai/fastbook/blob/master/16_accel_sgd.ipynb) | [Bölüm 17, Foundations](https://colab.research.google.com/github/fastai/fastbook/blob/master/17_foundations.ipynb) | [Bölüm 18, GradCAM](https://colab.research.google.com/github/fastai/fastbook/blob/master/18_CAM.ipynb) | [Bölüm 19, Learner](https://colab.research.google.com/github/fastai/fastbook/blob/master/19_learner.ipynb) | [Bölüm 20, conclusion](https://colab.research.google.com/github/fastai/fastbook/blob/master/20_conclusion.ipynb)


## Contributions / Katkılar

Bu Repo'ya herhangi bir Pull isteğinde bulunursanız bu çalışmanın telif hakkını Jeremy Howard ve Sylvain Gugger'a devretmiş olursunuz. (Ayrıca, yazım veya metin üzerinde küçük düzenlemeler yapıyorsanız, lütfen dosyanın adını ve neyi düzelttiğinizin çok kısa bir açıklamasını belirtin. İncelemeyi yapanların hangi düzeltmelerin daha önce yapıldığını bilmesi zordur. Teşekkür ederiz.)

## Atıf

Kitaba atıfta bulunmak istiyorsanız, şunu kullanabilirsiniz:

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


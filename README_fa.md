[English](./README.md) / [Spanish](./README_es.md) / [Korean](./README_ko.md) / [Chinese](./README_zh.md) / [Bengali](./README_bn.md) / [Indonesian](./README_id.md) / [Italian](./README_it.md) / [Portuguese](./README_pt.md) / [Vietnamese](./README_vn.md) / [Persian](./README_fa.md)

# کتاب fastai

این دفترچه‌ها مقدمه‌ای بر یادگیری عمیق، 
[fastai](https://docs.fast.ai/)
و
[PyTorch](https://pytorch.org/)
را پوشش می‌دهند.
 fastai یک رابط برنامه‌نویسی لایه لایه برای یادگیری عمیق است.
 برای اطلاعات بیشتر به
 [مقاله‌ی fastai](https://www.mdpi.com/2078-2489/11/2/108)
 مراجعه کنید.
 همه چیز در این مخزن شامل حق نشر جرمی هوارد و سیلوین گوگر از 2020 به بعد است.

این دفترچه‌ها برای یک درس آنلاین آزاد یا
[MOOC](https://course.fast.ai)
استفاده می‌شوند و پایه‌ی
[این کتاب](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527) 
را تشکیل می‌دهند،
که در حال حاضر برای خرید موجود است.
کتاب محدودیت‌های حق نشر همگانی یا
GPL
مشابهی که در این مخزن وجود دارد را ندارد.

کد موجود در دفترچه‌ها و فایل‌های پایتون
`py.`
 تحت مجوز
 GPL v3
 است؛
 برای جزئیات به فایل 
 LICENSE
 مراجعه کنید. 
 بقیه (شامل تمام نوشته‌های 
 markdown
 در دفترچه‌ها و غیره)
 برای هیچ گونه انتشار مجدد یا تغییر قالب یا انتشار در رسانه مجوز ندارند،
 مگر برای رونوشت از دفترچه‌ها یا جدا کردن این مخزن برای استفاده‌ی شخصی شما.
 استفاده‌ی تجاری یا انتشار رسانه‌ای مجاز نیست.
 ما این مطالب را به صورت رایگان در دسترس قرار می‌دهیم تا به شما در آموختن یادگیری عمیق کمک کنیم،
 بنابراین لطفاً به حق نشر و این محدودیت‌ها احترام بگذارید.

اگر می‌بینید کسی در جای دیگری یک نسخه از این مطالب را منتشر می‌کند، لطفاً به آن‌ها اطلاع دهید که کارشان مجاز نیست و ممکن است منجر به پیگرد قانونی شود.
به علاوه، رفتارشان به جامعه آسیب می‌زند، چراکه اگر مردم حق نشر ما را نادیده بگیرند، احتمالاً مطالب بیشتری را به این روش منتشر نمی‌کنیم.

## دفترچه‌ی Colab

به جای دریافت این مخزن و باز کردن آن در دستگاه خود، می‌توانید با استفاده از
[Google Colab](https://research.google.com/colaboratory/)
دفترچه‌ها را بخوانید و با آن کار کنید.
این رویکرد به خصوص برای افرادی که به تازگی شروع به کار کرده‌اند توصیه می‌شود
-- نیازی به راه‌اندازی یک محیط توسعه‌ی پایتون در دستگاه خود ندارید، زیرا می‌توانید مستقیماً در مرورگر وب خود شروع به کار کنید.

می‌توانید
با کلیک کردن روی این پیوندها هر فصلی از کتاب را در
Colab
باز کنید:
[آشنایی با ژوپیتر](https://colab.research.google.com/github/fastai/fastbook/blob/master/app_jupyter.ipynb) | [فصل 1، مقدمه](https://colab.research.google.com/github/fastai/fastbook/blob/master/01_intro.ipynb) | [فصل 2، ساخت](https://colab.research.google.com/github/fastai/fastbook/blob/master/02_production.ipynb) | [فصل 3، اخلاق](https://colab.research.google.com/github/fastai/fastbook/blob/master/03_ethics.ipynb) | [فصل 4، مبانی MNIST](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb) | [فصل 5، نژاد حیوان خانگی](https://colab.research.google.com/github/fastai/fastbook/blob/master/05_pet_breeds.ipynb) | [فصل 6، چند دسته](https://colab.research.google.com/github/fastai/fastbook/blob/master/06_multicat.ipynb) | [فصل 7، اندازه و زمان تست](https://colab.research.google.com/github/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb) | [فصل 8، پالایش گروهی](https://colab.research.google.com/github/fastai/fastbook/blob/master/08_collab.ipynb) | [فصل 9، داده‌ی جدولی](https://colab.research.google.com/github/fastai/fastbook/blob/master/09_tabular.ipynb) | [فصل 10، پردازش زبان طبیعی](https://colab.research.google.com/github/fastai/fastbook/blob/master/10_nlp.ipynb) | [فصل 11، رابط برنامه‌نویسی](https://colab.research.google.com/github/fastai/fastbook/blob/master/11_midlevel_data.ipynb) | [فصل 12، سیری در پردازش زبان طبیعی](https://colab.research.google.com/github/fastai/fastbook/blob/master/12_nlp_dive.ipynb) | [فصل 13، کانولوشن](https://colab.research.google.com/github/fastai/fastbook/blob/master/13_convolutions.ipynb) | [فصل 14، Resnet](https://colab.research.google.com/github/fastai/fastbook/blob/master/14_resnet.ipynb) | [فصل 15، جزئیات معماری](https://colab.research.google.com/github/fastai/fastbook/blob/master/15_arch_details.ipynb) | [فصل 16، بهینه‌سازی و بازخوانی](https://colab.research.google.com/github/fastai/fastbook/blob/master/16_accel_sgd.ipynb) | [فصل 17، زیربنا](https://colab.research.google.com/github/fastai/fastbook/blob/master/17_foundations.ipynb) | [فصل 18، GradCAM](https://colab.research.google.com/github/fastai/fastbook/blob/master/18_CAM.ipynb) | [فصل 19، Learner](https://colab.research.google.com/github/fastai/fastbook/blob/master/19_learner.ipynb) | [فصل 20، جمع‌بندی](https://colab.research.google.com/github/fastai/fastbook/blob/master/20_conclusion.ipynb)


## مشارکت‌ها

اگر هر گونه درخواست مشارکتی در این مخزن داشته باشید، حق نشر آن را به جرمی هوارد و سیلوین گوگر داده‌اید.
(به علاوه، اگر ویرایش‌های کوچکی در املا یا متن انجام می‌دهید،
لطفاً نام فایل را و توضیح بسیار مختصری از آنچه در حال رفعش هستید بیان کنید.
برای بازبینان سخت است که بدانند کدام اصلاحات قبلاً انجام شده است. ممنون.)

## استناد

اگر خواستید به کتاب استناد کنید، می‌توانید از کد زیر استفاده کنید:

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

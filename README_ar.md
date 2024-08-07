[English](./README.md) / [Spanish](./README_es.md) / [Korean](./README_ko.md) / [Chinese](./README_zh.md) / [Bengali](./README_bn.md) / [Indonesian](./README_id.md) / [Italian](./README_it.md) / [Portuguese](./README_pt.md) / [Vietnamese](./README_vn.md) / [Arabic](./README_ar.md)

# Fastai  كتاب


[PyTorch](https://pytorch.org/) و [fastai](https://docs.fast.ai/) تغطي هذه الدفاتر مقدمة للتعلم العميق

[The fastai paper](https://www.mdpi.com/2078-2489/11/2/108) عبارة عن واجهة برمجة متعددة الطبقات للتعلم العميق؛ لمزيد من المعلومات، راجع الورقة fastai
 
 بدءا من عام 2020 فصاعدا, حقوق الطبع والنشر لهذا المستودع ومحتوياته تعود إلى جيريمي هوارد وسيلفين غوججر. مجموعة مختارة من الفصول متاحة  [اقرأ على الانترنت هنا](https://fastai.github.io/fastbook2e/)

   يتم استخدام دفاتر الملاحظات الموجودة في هذا المستودع في [دورة مفتوحة المصدر](
https://course.fast.ai) وتشكل أساس هذا [الكتاب](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527) المتاح للشراء حاليًا 
 و الذي بدوره لا يخضع لنفس قيود الملكية الفكرية المطبقة على هذا المستودع

الكود الموجود في دفاتر الملاحظات وملفات البايثون يخضع لقوانين الرخصة العمومية الشاملة (الاصدار 3),  راجع ملف الترخيص للحصول على تفاصيل اكثر.
أما الباقي (بما في ذلك جميع خلايا النصوص في دفاتر الملاحظات وغيرها) فهي غير مرخصة لإعادة التوزيع أو تغيير في الشكل و النوع، بخلاف عمل نسخ من هذا المستودع لاستخدامك الشخصي.لايسمح أي استخدام تجاري لهذه المواد.هذا المستودع متاح مجانًا لمساعدتك على تعلم التعلم العميق، لذا يرجى احترام حقوق النشر الخاصة بنا

إذا رأيت شخصًا يستضيف نسخة من هذه المواد في مكان آخر ، فيرجى إعلامه بأن أفعاله غير مسموح بها وقد تؤدي إلى الملاحقة القانونية. علاوة على ذلك ، سوف تضر افعاله المنفردة بالمجتمع ككل لأننا من غير المحتمل أن ننشر موادا إضافية بهذه الطريقة (مجانا) إذا تجاهل الناس حقوق الطبع والنشر الخاصة بنا

## Colab  غوغل كولاب

 يمكنك القراءة والعمل مع دفاتر الملاحظات باستخدام [غوغل كولاب](https://research.google.com/colaboratory/) كبديل عوضا عن استنساخ هذا المستودع وفتحه على جهازك 

هذا هو الأسلوب الموصى به بالنسبة للأشخاص المبتدئين في هذا المجال حيث يمكنك العمل مباشرة في متصفح الويب الخاص بك و ليست هناك حاجة لإعداد بيئة تطوير بايثون على جهازك الخاص

 :يمكنك فتح أي فصل من الكتاب في منصة غوغل كولاب بالضغط على أحد هذه الروابط ادناه 

 [مقدمة إلى جوبيتر](https://colab.research.google.com/github/fastai/fastbook/blob/master/app_jupyter.ipynb)| [الفصل 1، المقدمة](https://colab.research.google.com/github/fastai/fastbook/blob/master/01_intro.ipynb) | [الفصل 2، من النموذج إلى الإنتاج](https://colab.research.google.com/github/fastai/fastbook/blob/master/02_production.ipynb) |[الفصل 3، أخلاقيات معالحة البيانات](https://colab.research.google.com/github/fastai/fastbook/blob/master/03_ethics.ipynb) |
 [الفصل 4، تدريب مصنف أرقام](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb)| [الفصل 5، تصنيف الصور](https://colab.research.google.com/github/fastai/fastbook/blob/master/05_pet_breeds.ipynb) | [الفصل 6، تصنيف الصور（تابع）](https://colab.research.google.com/github/fastai/fastbook/blob/master/06_multicat.ipynb) | [الفصل 7، تدريب نموذج حديث](https://colab.research.google.com/github/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb) | [الفصل 8 ،نبذة مفصلة في التصفية التعاونية](https://colab.research.google.com/github/fastai/fastbook/blob/master/08_collab.ipynb) | [الفصل 9، نبذة مفصلة في النمذجة الجدولية](https://colab.research.google.com/github/fastai/fastbook/blob/master/09_tabular.ipynb) |[الفصل 10، معالجة اللغات الطبيعية](https://colab.research.google.com/github/fastai/fastbook/blob/master/10_nlp.ipynb) | [الفصل 11، واجهة  برمجة التطبيقات للمستوى المتوسط](https://colab.research.google.com/github/fastai/fastbook/blob/master/11_midlevel_data.ipynb) | [الفصل 12،  نبذة مفصلة في معالجة اللغات الطبيعية](https://colab.research.google.com/github/fastai/fastbook/blob/master/12_nlp_dive.ipynb) | [الفصل 13، الشبكات العصبية التلافيفية ](https://colab.research.google.com/github/fastai/fastbook/blob/master/13_convolutions.ipynb) | [الفصل 14، ريسنيت](https://colab.research.google.com/github/fastai/fastbook/blob/master/14_resnet.ipynb) | [الفصل 15، هندسة التطبيقات، نبذة مفصلة](https://colab.research.google.com/github/fastai/fastbook/blob/master/15_arch_details.ipynb) | [الفصل 16، عملية التدريب](https://colab.research.google.com/github/fastai/fastbook/blob/master/16_accel_sgd.ipynb) | [الفصل 17، الشبكة العصبية من الأسس](https://colab.research.google.com/github/fastai/fastbook/blob/master/17_foundations.ipynb) | [الفصل 18، تفسير الشبكات العصبية](https://colab.research.google.com/github/fastai/fastbook/blob/master/18_CAM.ipynb) | [fastai الفصل 19، متعلم سريع من الصفر](https://colab.research.google.com/github/fastai/fastbook/blob/master/19_learner.ipynb) |

[الفصل 20، الخاتمة](https://colab.research.google.com/github/fastai/fastbook/blob/master/20_conclusion.ipynb)


## مساهمات 

إذا قمت بتقديم أي طلبات سحب لهذا المستودع، فإنك تقوم بتعيين حقوق الطبع والنشر لهذا العمل لجيريمي هوارد وسيلفان جوجر. (بالإضافة إلى ذلك، إذا كنت تقوم بإجراء تعديلات صغيرة على التهجئة أو النص، فيرجى تحديد اسم الملف وارفاق وصف موجز لما قمت بتعديله لتسهيل عمل المراجعين في معرفة التصحيحات التي تم إجراؤها بالفعل. شكرًا لك.)


## اقتباسات

:إذا أردت الاستشهاد بالكتاب يمكنك استخدام الصيفة الموضحة في الاسفل

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


# Questionar for lesson 1


1. Do you need these for deep learning?

Lots of math T / F
Lots of data T / F
Lots of expensive computers T / F
A PhD T / F



2. Name five areas where deep learning is now the best in the world.

`Natural language processing (NLP):: Answering questions; speech recognition; summarizing documents; classifying documents; finding names, dates, etc. in documents; searching for articles mentioning a concept
Computer vision:: Satellite and drone imagery interpretation (e.g., for disaster resilience); face recognition; image captioning; reading traffic signs; locating pedestrians and vehicles in autonomous vehicles
Medicine:: Finding anomalies in radiology images, including CT, MRI, and X-ray images; counting features in pathology slides; measuring features in ultrasounds; diagnosing diabetic retinopathy
Biology:: Folding proteins; classifying proteins; many genomics tasks, such as tumor-normal sequencing and classifying clinically actionable genetic mutations; cell classification; analyzing protein/protein interactions
Image generation:: Colorizing images; increasing image resolution; removing noise from images; converting images to art in the style of famous artists
Recommendation systems:: Web search; product recommendations; home page layout
Playing games:: Chess, Go, most Atari video games, and many real-time strategy games
Robotics:: Handling objects that are challenging to locate (e.g., transparent, shiny, lacking texture) or hard to pick up
Other applications:: Financial and logistical forecasting, text to speech, and much more...`

3. What was the name of the first device that was based on the principle of the artificial neuron?

`Perceptron`

4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?

`
A set of processing units
A state of activation
An output function for each unit
A pattern of connectivity among units
A propagation rule for propagating patterns of activities through the network of connectivities
An activation rule for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit
A learning rule whereby patterns of connectivity are modified by experience
An environment within which the system must operate
`

5. What were the two theoretical misunderstandings that held back the field of neural networks?




6. What is a GPU?

[GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit)


7. Open a notebook and execute a cell containing: 1+1. What happens?

2

8. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.

✅

9. Complete the Jupyter Notebook online appendix.

✅

10. Why is it hard to use a traditional computer program to recognize images in a photo?





11. What did Samuel mean by "weight assignment"?

`Weights are just variables, and a weight assignment is a particular choice of values for those variables.`


12. What term do we normally use in deep learning for what Samuel called "weights"?

`Paramiters`

13. Draw a picture that summarizes Samuel's view of a machine learning model.

✅

14. Why is it hard to understand why a deep learning model makes a particular prediction?



15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?

` A mathematical proof called the *universal approximation theorem* shows that this function can solve any problem to any level of accuracy, in theory.`


16. What do you need in order to train a model?


`When you train a model, you must _always_ have both a training set and a validation set, and must measure the accuracy of your model only on the validation set. If you train for too long, with not enough data, you will see the accuracy of your model start to get worse; this is called _overfitting_. fastai defaults `valid_pct` to `0.2`, so even if you forget, fastai will create a validation set for you!`



17. How could a feedback loop impact the rollout of a predictive policing model?

`This is a *positive feedback loop*, where the more the model is used, the more biased the data becomes, making the model even more biased, and so forth.
`


18. Do we always have to use 224×224-pixel images with the cat recognition model?

yes for historical reasons

19. What is the difference between classification and regression?

`Classification and Regression: classification and regression have very specific meanings in machine learning. These are the two main types of model that we will be investigating in this book. A classification model is one which attempts to predict a class, or category.`

20. What is a validation set? What is a test set? Why do we need them?

`This is absolutely critical, because if you train a large enough model for a long enough time, it will eventually memorize the label of every item in your dataset!`


21. What will fastai do if you don't provide a validation set?

`so even if you forget, fastai will create a validation set for you!`

22. Can we always use a random sample for a validation set? Why or why not?




23. What is overfitting? Provide an example.

`The training set does well but the test set does not do so well. we are able to correct this by adding more data if possible. or data augmentation, reduce the complexity, reducing the number of layers, this can also be known as dropout`

24. What is a metric? How does it differ from "loss"?

`The concept of a metric may remind you of *loss*, but there is an important distinction. The entire purpose of loss is to define a "measure of performance" that the training system can use to update weights automatically. In other words, a good choice for loss is a choice that is easy for stochastic gradient descent to use.`

25. How can pretrained models help?

`This can help based on the task, these models have already been traied to some extent, thus we use them for a type of transfer-learning`

26. What is the "head" of a model?

`The *head* of a model is the part that is newly added to be specific to the new dataset.`
`When using a pretrained model, `cnn_learner` will remove the last layer, since that is always specifically customized to the original training task (i.e. ImageNet dataset classification), and replace it with one or more new layers with randomized weights, of an appropriate size for the dataset you are working with. This last part of the model is known as the *head*.`

27. What kinds of features do the early layers of a CNN find? How about the later layers?

?

28. Are image models only useful for photos?

?

29. What is an "architecture"?

30. What is segmentation?

`Creating a model that can recognize the content of every individual pixel in an image is called segmentation.
`
31. What is y_range used for? When do we need it?

`This model is predicting movie ratings on a scale of 0.5 to 5.0 to within around 0.6 average error. Since we're predicting a continuous number, rather than a category, we have to tell fastai what range our target has, using the `y_range` parameter.`

32. What are "hyperparameters"?

`we are likely to explore many versions of a model through various modeling choices regarding network architecture, learning rates, data augmentation strategies, and other factors we will discuss in upcoming chapters. Many of these choices can be described as choices of *hyperparameters*.`

33. What's the best way to avoid failures when using AI in an organization?
`(It's also a good idea for you to try out some simple baseline yourself, so you know what a really simple model can achieve. Often it'll turn out that your simple model performs just as well as one produced by an external "expert"!)`
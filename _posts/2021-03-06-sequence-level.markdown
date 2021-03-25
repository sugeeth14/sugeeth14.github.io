---
layout: post
title:  "Understanding Word Level and Sequence Level Knowledge Distillation"
date:   2021-03-26 00:52:11 +0530
categories: jekyll update
---

<p align="center">
  <img src="https://raw.githubusercontent.com/sugeeth14/sugeeth14.github.io/sequence_level/images/Intro.jpeg" />
</p>
<!-- <p style="text-align: center;color: grey">credit: from Unsplash by eberhard grossgasteiger</p> -->
<br />

In this post we will be talking about **Sequence Level Knowledge Distillation** which is particularly useful for distilling *Sequence2Sequence* models used in prominent tasks like Translation, Grammar Correction, Speech to Text among others in Natural Language Processing.<br /><br />
<!-- 
We will also try to implement a Teacher Student model using fairseq to translate from English to German as we walk through. -->

### Intro: Why do we need Knowledge Distillation?

NLP models feed on huge training data. Typically for a Neural Machine Translation system to train, it takes atleast few millions of data to get a model output something reasonable. The top performing [WMT submissions](https://www.aclweb.org/anthology/2020.wmt-1.1.pdf) are often trained on hundreds of millions of data.  <br /> <br />  

<p align="center">
  <img src="https://raw.githubusercontent.com/sugeeth14/sugeeth14.github.io/sequence_level/images/feed.gif" />
</p>
<p style="text-align: center;color: grey">Feed me moreeee **data**</p>
<br />

But with great data come in great parameters, to learn the data's distribution, large scale model with billions of parameters are needed. For example [GPT-3](https://arxiv.org/pdf/2005.14165.pdf) was trained with 175 billion parameters !!! With such huge parameters, it is difficult to deploy models in production system and on top of that, time to process an input text increases exponentially. <br /><br />

<p align="center">
  <img src="https://raw.githubusercontent.com/sugeeth14/sugeeth14.github.io/sequence_level/images/fat.gif" />
</p>
<p style="text-align: center;color: grey">How do I move now !!?</p>
<br />


### Knowldege Distillation
Here comes our saviour - **Knowledge Distillation**: also called **Teacher-Student Network** is a very effecient method which comes in handy in cases when the model size has to be reduced or the model's latency has to be reduced.

There are many awesome posts which have discussed Knowledge Distillation and the underlying concepts. I'll try to link some of them and possibly will write an exclusive blog on Knowledge distillation myself.

#### The Basic idea

To understand the basic idea of Knowledge Distillation, I will try to illustrate it with an example.

<p align="center">
  <img src="https://raw.githubusercontent.com/sugeeth14/sugeeth14.github.io/sequence_level/images/galaxy.jpeg" />
</p>
<!-- <p style="text-align: center;color: grey">How do I move now !!?</p> -->
<br />

Let's say you want to understand the essense of of ***General Theory of Relativity***, there are two ways to try to understand and come to a conclusion about the theory. 

 
1. You could simply read Einstein's thesis and a large chunk of not so cool equations and try understand what it is and break your mind. 

2. You could just straight away look up to the cosmos and heavenly bodies and try to understand why they behave the way the are behaving and revolving around heavenly bodies in the fabric of space. And, come up with the theory and equations yourself.<br /><br />


<p align="center">
  <img src="https://raw.githubusercontent.com/sugeeth14/sugeeth14.github.io/sequence_level/images/einstein.gif" />
</p>
<p style="text-align: center;color: grey">seems simple ehh ?</p>
<br />

 
As you might have guessed already the first alternative is easier to do. This is essentially how **Knowledge Distillation** works. 

The teacher model or the model which has considerably larger parameters tries to learn and map the distribution of the target space and stores its "learning" in the form of weights of the network. Now, the student model which is small in parameters tries to learn the learning of the teacher by trying to map the distribution the teacher has learned instead of learning the original distribution.

Predominantly Knowledge Distillation has been used over time in Classification tasks (though primarily all sequence tasks too boil themselves down to classification). The Knowledge Distillation we are going to specifically talking about today is Sequence Level Knowledge Distillation and we'll try to understand it in the context of **Neural Machine Translation**.

### What does Machine Translation actually try to do ?

First we will try to understand what Machine Translation tries to accomplish when translating from English to German.

<!-- Consider the following example which we want to translate from English to German.

<br />

>English: I love to play games.  
German: Ich liebe es, Spiele zu spielen.

<br />


Let $$v = [v_{1}, v_{2}, v_{3}, ..., v_{l}]$$ be the total unique words present in *German* language, and $$l$$ be the length of this vocabulary.

Let's do a calculation now: how many unique sentences can we construct in German, which have 3 words in them ?

Number of German sentences with 3 words  $$l$$ $$\times$$ $$l$$ $$\times$$ $$l$$

and how many with 4 words ?

Number of German sentences with 4 words = $$l$$ $$\times$$ $$l$$ $$\times$$ $$l$$ $$\times$$ $$l$$

But if the length of the German sentence increases, this value grows exponentially. But since we don't know the length of translated sentences until we translate, there are quite a large number of possible ways we can arrange the words in a sequence. But just one (or probably a few depending on the language) of these sequences is the right translation of the source sentence. In this case the sentence

<br />

>Ich liebe es, Spiele zu spielen.

<br /> -->

If $$\tau$$ is the set of all possible sequences in German language and $$t$$ is the translated sequence we are aiming to find and $$s$$ is the source sentence, MT system tries to find the sequence $$t \in \tau$$ which has the highest probability of occurence give the source sentence.

$$argmax_{t \in \tau}\,p(t \mid s)$$




### Word Level Knowledge Distillation


Neural Machine Translation systems are trained to minimize word level Negative Log Likelihood loss.

That is at every position we compute the following loss with the target word

$$L_{NLL}(\theta) = -\sum_{ k = 1}^{|\nu|}1{\{y = k\}} \log p(y = k \mid x;\theta)$$ 


But in case of word level KD, we are no longer trying to match the distribution of ground truth but rather the distribution of teacher. The point is we try to train to match student's predictions with teacher's predictions. So at any position we compute the loss for that position alone using


$$L_{NLL}(\theta) = -\sum_{ k = 1}^{|\nu|}q(t_{j} = k \mid s,t_{<j}) \log p(y = k \mid x;\theta)$$ 

where $$q(y \mid x; \theta_{T})$$ is the distribution of the teacher.


<p align="center">
  <img src="https://raw.githubusercontent.com/sugeeth14/sugeeth14.github.io/sequence_level/images/word-level.png" />
</p>
<p style="text-align: center;">Word level KD</p>

As seen above the prediction of teacher(yellow) vector is used to train the student instead of ground truth.


### Sounds good but why Sequence Level KD ?

Sequence Level Knowledge Distillation was introduced in the paper titled [Sequence-Level Knowledge Distillation](https://arxiv.org/pdf/1606.07947.pdf) by Yoon Kim and Alexander M. Rush from Harvard University.

The core idea is this 

>*"Word-level knowledge distillation allows transfer of
these local word distributions. Ideally however, we
would like the student model to mimic the teacher’s
actions at the sequence-level."*

Let us try to decipher this sentence, which has the core idea of Sequence Level KD.

Let us say we are trying to translate the following sentence from English to German.

>English: I love to play games.  
German: Ich liebe es, Spiele zu spielen.

Unlike a Cat - Dog classifier or any other multilabel classifier Translation is a sequence 2 sequence task. In other words, there will be dependencies between different words to be translated and they are not just isolated classification tasks.

The above Word-level KD allows transfer of local word level distributions from teacher to student but ideally student should learn the distribution of teacher on a larger sequence scale.

This is when sequence level KD comes into play.

<p align="center">
  <img src="https://raw.githubusercontent.com/sugeeth14/sugeeth14.github.io/sequence_level/images/Sequence-level.png" />
</p>
<p style="text-align: center;">Sequence Level KD</p>

When a wrong prediction is made at a particular word, it propagates forward so weighing the loss with respect to the whole sequence is important.

As shown above student is trained on whole teacher sequence instead of word by word loss computation. The loss is now computed on the whole Sequence.

$$L_{SEQ-KD} = - \sum_{t \in T}q(t \mid s) \log p(t \mid s)$$

where $$q(t \mid s)$$ is the distribution of the teacher

<!-- $$L_{SEQ-KD} = - \sum_{t belongs to T}q(t_{j} = k \mid s,t_{ <j }) *  \log p(t_{j} = k \mid s,t_{<j})$$ -->

We can replace the teachers distribution of $$q(t \mid s)$$ by approximating it with the mode of the parameters.

The mode of the parameters is where the sentence with highest probability score of the teacher parameters.

Ideally we must have   **beam size = size(vocabulary)** to find this mode of teacher distribution.
But for simplicity and to not go into realm of computationally not feasible areas, we use beam size to approximate for the mode.
 
let $$\hat{y}$$ be such a sequence found by beam searching the teacher now the sequence level KD is as below.

$$L_{SEQ-KD} ≈ −\sum_{t \in T} 1 \{t = \hat{y}\} \log p(t \mid s) 
             = − \log p(t = yˆ \mid s)$$

#### Why is this a good approximation ? 

This is because large portion of $$q$$'s mass lies in single sequence.

### In summary this is what we do 
1. Train teacher 
2. Infer the training set using teacher. 
3. Discard the original ground truth sequences and replace with teacher inferences
4. Train the student.

<!-- $$L_{SINGLE-WORD-KD} = -\sum_{ k = 1}^{|\nu|}q(t_{j} = k \mid s,t_{ <j }) *  \log p(t_{j} = k \mid s,t_{<j})$$

Now we sum this value over the entire sequence to get the total Loss.

$$L_{WORD-KD} = -\sum_{j = 1}^{J}\sum_{ k = 1}^{|\nu|}q(t_{j} = k \mid s,t_{ <j }) *  \log p(t_{j} = k \mid s,t_{<j})$$ -->



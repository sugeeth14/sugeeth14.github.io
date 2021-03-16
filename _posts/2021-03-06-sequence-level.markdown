---
layout: post
title:  "Understanding Sequence Level Knowledge Distillation : Hands on with Fairseq"
date:   2021-03-06 23:50:11 +0530
categories: jekyll update
---

In this post we will be talking about Sequence Level Knowledge distillation which is particularly useful for distilling Sequence Level models used in prominent tasks like Translation, Grammar Correction etc in Natural Language Processing.

We will also try to implement a Teacher Student model using fairseq to translate from English to German as we walk through.

Knowledge Distillation is a very effecient method which comes in handy in cases when the model size has to be reduced or the model's latency has to be reduced.

There are many blogs which have discussed Knowledge Distillation and the underlying concepts. I'll try to link some of them and possibly will write an exclusive blog on Knowledge distillation myself.

To understand the basic idea of Knowledge Distillation, I will try to illustrate it with an example.

Let's say you want to understand the essense of of General Theory of Relativity, there are two ways to try to understand and come to a conclusion about the theory. You could just straight away look up to the cosmos and heavenly bodies and try to understand why they behave the way the are behaving and revolving around heavenly bodies in the fabric of space. Or, you could simply read and try work your way on the theory or relativity introduced by Einstein. This is essentially how Knowledge Distillation works. The teacher model or the model which has considerably larger parameters tries to learn and map the distribution of the target space and stores its "learning" in the form of weights of the network. Now, the student model which is small in parameters tries to learn the learning of the teacher by trying to map the distribution the teacher has learned instead of learning the original distribution.


Predominantly Knowledge Distillation has been used over time in Classification tasks ( though primarily all sequence tasks too boil themselves down to classification) The Knowledge Distillation we are going to specifically talking about today is Sequence Level Knowledge Distillation.

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


### What does Machine Translation actually try to do ?

First we will try to understand what Machine Translation tries to accomplish.

Consider the following example which we want to translate from English to German.

>English: I love to play games.  
German: Ich liebe es, Spiele zu spielen.

Let v = [v1, v2, v3, ...] be the total unique words present in German language, and L be the length of this vocabulary.

Now how many unique sentences can we construct which have 3 words in them.

It will be L * L * L

how many with 4 words

It will be L * L * L * L

But if the length of the German sentence is unknown essentially there are infinite possible ways we can arrange the words in a sequence.

But just one ( or probably a few depending on the language) of these sequences is the right translation of the source sentence.

We want to find such a sentence or the most probable sentence give the source. This is what machine Translation tries to find.

$$argmax_{t \in \tau}\,p(t \mid s)$$

Where tau is the set of all possible sequences.


### Word Level Knowledge Distillation

First, let us try to understand how word level Knowledge Distillation works here.

Neural Machine Translation systems are trained to minimize word level Negative Log Likelihood loss 

$$L_{NLL}(\theta) = -\sum_{ k = 1}^{|\nu|}1{\{y = k\}} \log p(y = k \mid x;\theta)$$ 

at each position or time step of the output.

The point is we try to train to match student's predictions with teacher's predictions.

<p align="center">
  <img src="images/word-level.jpeg" />
</p>
<p style="text-align: center;">*Fig. 2*</p>




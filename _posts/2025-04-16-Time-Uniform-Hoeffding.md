---
title: 'Time Uniform Hoeffding Inequality'
date: 2025-04-16
permalink: /posts/2025/04/time-uniform-hoeffding/
tags:
  - Statistics
  - Concentration Inequalities
  - Anytime Valid Inference
excerpt: ''
authors: "<a href='https://harit7.github.io/'>Harit Vishwakarma</a>" 
---

Suppose we have a coin with bias $p$. We toss it over time and each time we observe $X_t$, which is $1$ with probability $p$ and $0$ with probability $1-p$. Our running estimate of the bias is 
$$\hat{p}_t = \frac{1}{t} \sum_{i=1}^t X_t$$. We wish to get a confidence interval $\psi(t,\delta)$ on $\hat{p}_t$ that is valid *simultaneously* for all $t>0$. 

$$P \Big( \forall t >0 \quad |\hat{p}_t - p| \le \psi(t,\delta) \Big) \ge 1-\delta$$


Lets see how we can apply Hoeffding's inequality here.

**Attempt 1: Naive application.** The canonical [Hoeffding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality)  applies to a single estimate, i.e. we can have the following guarantee *for some specified $t$* by direct application,

$$P \Big( \quad |\hat{p}_t - p| \le \psi_1(t,\delta) \Big) \ge 1-\delta,$$

where $$\psi_1(t,\delta) = \sqrt{\frac{1}{2t} \log(\frac{2}{\delta})}.$$

So, it doesn't give us what we wanted!

**Attempt 2: What if we know we will stop at time $T$.**
This is interesting, when we need the guarantee for all $0<t\le T$, then we can invoke the above inequality for each $0<t\le T $ but with failure probability $\frac{\delta}{T}$. The union bound will ensure the overall failure probability is bounded by $\delta$. More specifically, we will get the following,

$$P \Big( \quad |\hat{p}_t - p| \le \psi_2(t,\frac{\delta}{T}) \Big) \ge 1-\delta,$$

where, $$\psi_2(t,\delta) = \psi_1(t, \frac{\delta}{T})  = \sqrt{\frac{1}{2t} \log(\frac{2T}{\delta})}.$$

OK, we have made some progress with union bound. Our new confidence interval $\psi_2(\delta, t)$ is simultaneously valid for a range of $0<t \le T$. But we are still far from what we originally wanted. Recall, we want $\psi(t,\delta)$ that is simultaneously valid for all $t>0$.

**Attempt 3: Naive generalization of Attempt 2.**
Caution: It is tempting to make a mistake and apply the above bound by replacing $T$ with $t$. Lets see how will this unfold: we are saying, $\psi(t,\delta) = \sqrt{\frac{1}{2t} \log(\frac{2t}{\delta})}$ is valid simultaneously for all $t>0$. It is not hard to see that $\psi(t,\delta) \to 0$ as  $t\to \infty$. So this bound is sensible, however there is a big issue. The overall failure probability is no longer bounded. How? 

Essentially, for this bound we are hoping that each $\psi(t,\delta)$ is not valid (fails) with probability $\frac{\delta}{t}$. Thus the overall failure probability for all $t>0$,


$$\delta + \frac{\delta}{2} + \ldots + \frac{\delta}{t} + \ldots = \delta \sum_{t=1}^\infty \frac{1}{t}.$$ 


Surprisingly, the sum $\sum_{t=1}^\infty \frac{1}{t}$ diverges. See [Harmonic series on wikipedia](https://en.wikipedia.org/wiki/Harmonic_series_(mathematics)).

**Attempt 4: The magic trick.**
The previous attempt was quite promising but did not work and it seems we might have hit a wall in trying to obtain our desired bound using simple union bound on Hoeffding's inequalty. If you have followed it so far, hold on for 1 sec we are not that far!! We can make our Attempt 3 work.
The trick is to bound the failure probability at each $t$ by $\frac{\delta}{t^2}$. With this, our overall failure probability is,



$$\delta + \frac{\delta}{4} + \ldots + \frac{\delta}{t^2} + \ldots = \delta \sum_{t=1}^\infty \frac{1}{t^2} = \delta\frac{\pi^2}{6}.$$ 

Detour: Finding the summation $\sum_{t=1}^\infty \frac{1}{t^2}$ exactly is known as the [Basel Problem](https://en.wikipedia.org/wiki/Basel_problem). Thanks to Euler, we don't have to solve this summation. He showed that it is equal to $\pi^2/6$.

This is exciting! we finally have a sequence of failure probabilities, in other words a sequence of confidence intervals that works simultaneously for all $t>0$. How does our $\psi(t,\delta)$ look like now, 
$$\psi(t,\delta) = \sqrt{\frac{1}{2t} \log(\frac{12 t^2}{\pi^2 \delta})} = \sqrt{\frac{1}{t}\log(\frac{12t}{\pi^2 \delta })}$$


$$P \Big( \forall t >0 \quad |\hat{p}_t - p| \le \psi(t,\delta) \Big) \ge 1-\delta$$

where $\psi(t,\delta) = \sqrt{\frac{1}{t}\log(\frac{12t}{\pi^2 \delta })} < \sqrt{\frac{1}{t}\log(\frac{2t}{ \delta })}$

There we go! we have a nice simple time uniform Hoeffding bound that we were looking for.

**Remark 1.** The same technique can be applied to more general concentration inequalities e.g., Azuma-Hoeffding, DKW etc. to obtain their time uniform versions.

**Remark 2.** Note, while this is simple technique to get time uniform bounds. The bounds obtained are not tight. There are tighter bounds possible with scaling of $\sqrt{\log(\log(t))}$ instead of $\sqrt{\log(t)}$. 

**Remark 3.** We tried $\zeta(p) := \sum_{t=1}^\infty \frac{1}{t^p}$ with $p=1$ and $2$ above. We saw with $p=1$ the sum diverges, and with $p=2$ the sum is a nice little constant. Note, with $p=2$ the failure probabilities decrease quadratically with $t$, i.e. the confidence intervals get much wider over time (much wider than necessary indeed). How about $p \in (1,2)$? The function $\zeta(p)$ is called the [Riemann Zeta Function](https://mathworld.wolfram.com/RiemannZetaFunction.html). The function decreases rapidly in this range and it would be interesting to see how our confidence intervals will look like with $p \in (1,2)$. 

The next post will explore some of these remarks. 

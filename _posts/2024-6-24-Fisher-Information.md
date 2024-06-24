---
layout: post
title: Notes on Fisher Information Matrix and related identities
---

The Hessian of the KL divergence is the fisher information matrix

$$F_\theta = \nabla^2_u \text{D}_\text{KL}(p_\theta || p_u)|_{u=\theta} $$

where 
$$\text{D}_\text{KL}(p_\theta || p_u) = \mathbb E_{x \sim p_\theta} \left[ \log p_\theta(x) - \log {p_u(x)} \right]$$

So we have

$$\begin{align} F_\theta &= \nabla^2_u \text{D}_\text{KL}(p_\theta || p_u)|_{u=\theta} \\ &= \left[ \nabla_u^2 \int (\log p_\theta(x) - \log p_u(x))p_\theta(x) dx \right]_{u=\theta} \\ &= \int \left[ \nabla_u^2(\log p_\theta(x) - \log p_u(x))p_\theta(x) \right]_{u=\theta}dx \\ &= \int - p_\theta(x)(\nabla_\theta^2 \log p_\theta(x)) dx \\ &= \int -p_\theta(x)(\frac{p_\theta(x)\nabla^2_\theta p_\theta(x) - (\nabla_\theta p_\theta(x))(\nabla_\theta p_\theta(x))^\top}{p_\theta(x)^2}) dx \\ &= \int \frac{(\nabla_\theta p_\theta(x))(\nabla_\theta p_\theta(x))^\top}{p_\theta(x)^2} p_\theta(x) dx - \int \nabla^2_\theta p_\theta(x) dx \\ &= \int (\nabla_\theta \log p_\theta(x))(\nabla_\theta \log p_\theta(x))^\top p_\theta(x) dx - \nabla^2_\theta \int p_\theta(x)dx \\ &= \mathbb E_{x \sim p_\theta} \left[ (\nabla_\theta \log p_\theta(x))(\nabla_\theta \log p_\theta(x))^\top\right] \\ &= \text{Cov}_{x\sim p_\theta} (\nabla_\theta \log p_\theta(x)) \end{align}$$

- where we assume that $$p_u$$ is $$\mathcal C^2$$

## Exponential families

$$p_\eta(x) = \frac{h(x)}{\mathcal Z(\eta)} \exp(\eta^\top t(x))$$

Where $$\mathcal Z(\eta) = \int h(x) \exp(\eta^\top t(x)) dx$$

The moments are given by 

$$\xi = \mathbb E_{x \sim p_\eta} [t(x)]$$

#### 1 Formula for the moments

$$\begin{align} \xi &= \nabla_\eta \log \mathcal Z(\eta) \\ &= \frac{\nabla_\eta \mathcal Z(\eta)}{\mathcal Z(\eta)} \\ &= \frac{1}{\mathcal Z(\eta)} \int \nabla_\eta(h(x) \exp(\eta^\top t(x)))dx \\ &=  \int \frac{h(x) \exp(\eta^\top t(x))}{\mathcal Z(\eta)} t(x) dx \\ &= t(x) p_\eta(x) dx \\ &= \mathbb E_{x \sim p_\eta}[t(x)] \end{align}$$

#### 2 Log-likelihood

$$ \begin{align} l(\eta) &= \sum_{i=1}^n \log p_\eta(x) \\ &= \sum_{i=1}^n \left[ \eta^\top t(x_i) - \log \mathcal Z(\eta) \right] \end{align}$$


$$\begin{align} \nabla_\eta l(\eta) &= \hat \xi - \xi \\ \hat \xi &= \frac{1}{n} \sum_{i=1}^n t(x_i) \end{align}$$


#### 3 Fisher information matrix

Since $$\nabla_\eta \log p_\eta(x) = t(x) - \xi$$, we have

$$F_\eta = \text{Cov}_{x\sim p_\eta}(t(x))$$

#### 4 KL divergence

$$\begin{align}
  F_\eta &= \nabla_{\eta_2}^2 \text{D}_\text{KL}(p_{\eta_1} || p_{\eta_2})|_{\eta_2 = \eta_1} \\ &= \nabla_{\eta_2}^2 \mathbb E_{x \sim p_{\eta_1}} \left[ \log p_{\eta_1 }(x) - \log p_{\eta_2}(x) \right]|_{\eta_2 = \eta_1} \\ &= \nabla^2_{\eta} \log \mathcal Z(\eta)
\end{align}$$


# Machine Learning 2019 Spring - Notes

<h6 style="text-align: right">Instructor: Hung-Yi Lee</h6>
<h6 style="text-align: right">Note By: Wu-Jun Pei(B06902029)</h6>

## Links

-   [Course Website](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)
-   [Youtube Channel](https://www.youtube.com/playlist?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)

## Lecture 2 - Where Does The Error Come From

-   The error comes from bias and variance.
    -   Simple Model $\Leftrightarrow$ Small Variance
    -   Complicated Model $\Leftrightarrow$ Small Bias

-   Trade-off between bias and variance
    -   

## Lecture 3 - Gradient Descent

### Notations

-   L, Loss: Loss function
-   $\theta$: Parameters

$\theta^{t + 1} = \theta^{t} - \eta \nabla L(\theta^t)$

### Tips

#### Tuning Learning Rates

- Fixed-rated

    - Visualize *Loss* to *# of updates*

- Adaptive Learning Rates
  - Reduce $\eta$ by some factor every few epochs
  - e.g. $\eta^{t} = \frac{\eta}{\sqrt{t + 1}}$

- Adagrad

  - Divide the learning rate of **each parameter** by the **RMS of its previous derivatives**.

  - For a single parameter $w$
      $$
      w^{t + 1} \leftarrow w^{t} - \frac{\eta^{t}}{\sigma^{t}} g^t
      $$

      -   $\sigma^{t}​$: **RMS** of previous derivatives of parameter $w​$.
          $$
          \sigma^t = \sqrt{\frac{1}{t + 1}\sum_{i = 0}^t (g^i)^2}
          $$

      -   $g^{t} = \frac{\part L(\theta^t)}{\part w}$.

  - Or briefly
      $$
      w^{t + 1} \leftarrow w^{t} - \frac{\eta}{\sqrt{\sum_{i = 0}^t (g^i)^2}} g^t
      $$

  -   We can observe that the best step is $\frac{|\text{First Derivative|}}{\text{Second Derivative}}​$. And the denominator term is somehow going to represent the second derivative. 

#### Stochastic Gradient Descent

-   Pick an example $x^{n}​$, consider gradient about data $n​$.

#### Feature Scaling

-   Make different features have the same scaling.

-   For each dimension $i$, calculate mean $m_i$ and standard deviation $\sigma_i$, let
    $$
    x_i^r \leftarrow \frac{x_i^r - m_i}{\sigma_i}
    $$

### Extra Reading

-   [Reference](https://medium.com/雞雞與兔兔的工程世界/機器學習ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db)
-   [Reference](http://ruder.io/optimizing-gradient-descent/index.html)

#### Momentum

-   Concept of momentum, accelerates in relevant direction

-   Iterative step:
    $$
    \begin{align*}
    v_t & \leftarrow \beta v_{t - 1} - \eta \frac{\part L}{\part W} \\
    w & \leftarrow w + v_t
    \end{align*}
    $$

-   $\beta$ is often set to `0.9`

#### Adam

-   Combination of *Adagrad* and *Momentum*

-   Iterative step:
    $$
    \begin{align*}
    m_t & \leftarrow\beta_1 m_{t - 1} + (1 - \beta_1) \frac{\part L}{\part W} \\
    v_t & \leftarrow \beta_2 v_{t - 1} + (1 - \beta_2) (\frac{\part L}{\part W})^2 \\
    
    \hat{m_t} &= \frac{m_t}{1 - \beta^t_1} \\
    \hat{v_t} &= \frac{v_t}{1 - \beta^t_2} \\
    
    w_{t + 1} &\leftarrow w_t - \frac{\eta}{\sqrt{\hat{v_t}} + \epsilon} \hat{m_t}
    \end{align*}
    $$

-   $\beta_1, \beta_2, \epsilon$ is often set to `0.9, 0.9999, 1e-8` respectively.

## Lecture 4 - Classification

### Ideal Alternatives

-   Function (Model)
    $$
    f(x) = 
    \begin{cases}
    g(x) \ge 0.5, &\text{class 1} \\
    else, &\text{class 2}
    \end{cases}
    $$

-   Loss Function
    $$
    L(f) = \sum_{n} \delta(f(x^n) \ne \hat y^n)
    $$
    

-   Find the best function

    -   Perceptron, SVM

### Gaussian Distribution

$$
f_{\mu, \Sigma}(x) = \frac{1}{(2 \pi)^{\frac{D}{2}}} \frac{1}{|\Sigma|\frac{1}{2}} \exp(-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x - \mu))
$$

-   Input: vector $x​$; Output: Probability of sampling $x​$.

-   The shape of the function determines by *mean* $\mu​$ and *convariance matrix* $\Sigma​$.

-   Likelihood
    $$
    L(\mu, \Sigma) = \prod_n f_{\mu, \Sigma}(x^n)
    $$

-   Maximum Likelihood
    $$
    \mu^*, \Sigma^* = \arg\max_{\mu, \Sigma} L(\mu, \Sigma)
    $$

    -   $\mu^* = \frac{1}{n} \sum_n x^n$
    -   $\Sigma^* = \frac{1}{n}(x^n - \mu^*)(x^n - \mu^*)^T$

-   Modifying Model

    -   👎: Giving each Gaussian a covariance matrix increases the parameters of the model, may resulting in overfitting.
    -   👉: Share a covariance matrix among the Gaussian distributions.
    -   $\mu_1, \mu_2$ remains the same, $\Sigma = \Sigma^1 + \Sigma^2$. (Reference: Bishop, Ch 4.2.2)
    -   The model becomes linear.

-   Probability Distribution

    -   For binary features, we may assume they are from *[Bernoulli Distributions](https://zh.wikipedia.org/wiki/伯努利分布)*
    -   If we assume all the dimensions are independent, then using *Naive Bayes Classifier*.

### Posterior Probability

Rewrite $P(C_1|x)$ as
$$
\begin{align*}
P(C_1 | x)
&= \frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1) + P(x|C_2)P(C_2)} \\
&= \frac{1}{1 + \frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}} \\
&= \frac{1}{1 + \exp(-z)} = \sigma(z)
\end{align*}
$$
, with $z = \ln \frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}$. We can derive the sigmoid function $\sigma(z)$. And we can rewrite $z$ as
$$
\begin{cases}
z &= (\mu^1 - \mu^2)^T \Sigma^{-1}x &\text{($\mathbf{w}^Tx$)}\\
& -\frac{1}{2}(\mu^1)^T(\Sigma^1)^{-1}\mu^1 + \frac{1}{2}(\mu^2)^T(\Sigma^2)^{-1}\mu^2 + \ln \frac{N_1}{N_2}&\text{bias}
\end{cases}
$$
Thus, $P(C_1|x) = \sigma(wx + b)​$

## Lecture 5 - Logistic Regression

1.  Function Set
    $$
    \begin{align*}
    f_{w, b}(x) &= P_{w, b}(C_i | x) \\
    &= \sigma(\sum_{i}w_ix_i + b)
    \end{align*}
    $$

2.  Goodness of a Function

    -   Likelihood
        $$
        L(w, b) = \prod_{i} (f_{w, b}(x^i) \text{ if $\hat y^i \in C_1$ else } 1 - f_{w, b}(x^i))
        $$

    -   Finding a $w^*, b^* = \arg\max_{w, b} L(w, b)$ is equivalent to finding $w^*, b^* = \arg \min_{w, b} -\ln L(w, b)$

    -   Redefine $\hat y^n = 1 \text{ for class 1 else }  0$.
        $$
        \begin{align*}
        - \ln L(w, b) = \sum_n -[\hat y^n \ln f_{w, b}(x^n) + (1 - \hat y) \ln(1 - f_{w, b}(x^n))]
        \end{align*}
        $$

    -   Cross entropy between two Berboulli distribution

        -   Distribution p: $p(x = 1) = \hat y^n​$, $p(x = 0) = 1 - \hat y^n​$
        -   Distribution q: $q(x = 1) = f(x^n)$, $p(x = 0) = 1 - f(x^n)$
        -   Cross entropy between p, q $H(p, q)$
        -   Cross entropy $\rightarrow 0$ if the two distribution are similar.

    -   Loss function
        $$
        \begin{align*}
        L(f) &= \sum_nC(f(x^n), \hat y^n)
        \end{align*}
        $$
        , where $C​$ is cross entropy.

3.  Find the best function
    $$
    \begin{align*}
    \frac{\part\ln f_{w, b}(x)}{\part w_i} &=
    \sum_n -(\hat y - f_{w, b}(x)) x_i
    \end{align*}
    $$

    -   $\frac{\part \sigma(z)}{z} = \sigma(z) (1 - \sigma(z))$

### Discriminative v.s. Generative

-   They share the same function set if the covariance matrix is shared among the distributions.
-   Discriminative: $P(C_1 | x)$, find $\mathbf w, b$ with logistic regression.
-   Generative: $\sigma(\mathbf w x + b)$, find $\mu^1, \mu^2, \Sigma^{-1}$.
    -   $\mathbf w^T = (\mu^1 + \mu^2)^T \Sigma^{-1}$
    -   $b = \frac{-1}{2}(\mu^1)^T(\Sigma)^{-1}\mu^1 + \frac{-1}{2}(\mu^2)^T(\Sigma)^{-1}\mu^2 + \ln \frac{N_1}{N_2}​$
-   ==The same model, but different function is selected by the same training data.== Since we've made assumption on *generative models* to have Gaussian distribution.
-   Benefit of Generative Model
    -   Less training data is needed
    -   More robust to the noice

### Multi-class Classification

>   Bishop P.209 - 210

Assume there are $m$ classes, for each class $i$ with $\mathbf w^i, b_i$, calculate $z_i = \mathbf w^i \cdot x + b_i$

-   Softmax
    -   $T = \sum_i e^{z_i}$
    -   For each class $i$, $y_i = e^{z_i} / T$
    -   Maximum entropy
-   Cross Entropy
    -   $\sum_{i = 1}^{m} \hat{y_i} \ln y_i$

### Limitation of Logistic Regression

-   The boundary is a line
    -   Solution: Feature transformation, not good enough since we don't know howto

## Lecture 6 - Brief Introduction of Deep Learning

### Three Steps for Deep Learning

1. Neural network structure (a function set)
  - Deep = *Many hidden layers*
  - Given $\mathbf{x}^n$, output $\mathbf{y}^n$, via NN with parameters $\theta$.
2. Goodness of function
  - Cross Entropy
  - $L(\theta) = \sum_{n = 1}^N C^n(\theta)$
3. Pick the best function
    -   Based on gradient descent
    -   Backpropagation

## Lecture 7 - Backpropagation

It's simply *gradient descent*

#### Forward Pass

Calculate each output of neuron.

#### Backward Pass

Calculate each partial derivative from back to front.



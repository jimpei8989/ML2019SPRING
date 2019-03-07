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




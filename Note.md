# Machine Learning 2019 Spring - Notes

<h6 style="text-align: right">Instructor: Hung-Yi Lee</h6>
<h6 style="text-align: right">Note By: Wu-Jun Pei(B06902029)</h6>

## Links

-   [Course Website](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)
-   [Youtube Channel](https://www.youtube.com/playlist?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)

## Section 3 - Gradient Descent

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

-   Pick an example $x^{n}$, consider gradient about data $n$.

#### Feature Scaling

-   Make different features have the same scaling.

-   For each dimension $i$, calculate mean $m_i$ and standard deviation $\sigma_i$, let
    $$
    x_i^r \leftarrow \frac{x_i^r - m_i}{\sigma_i}
    $$


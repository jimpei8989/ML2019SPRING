# Machine Learning 2019 Spring - HW1

###### By: Wu-Jun Pei(B06902029)

### [Kaggle Website](https://www.kaggle.com/c/ml2019spring-hw1)

### 2/21

#### 1st Attempt

Linear Regression with Gradient Descent

- Parameters
    - Learning Rate $\eta$: `1e-8`
    - Iteration Time: `1e5`
- Features: PM2.5 values of previous 9 hours (without bias $x_0$)
- Optimizer: None
- Score = `5.99011`

#### 2nd Attempt

Linear Regression with Gradient Descent

- Parameters
    - Learning Rate $\eta$: `1e-8`
    - Iteration Time: `1e6`
- Features: PM2.5 values of previous 9 hours (with bias $x_0$)
- Optimizer: None
- Score = `5.93022`

### 2/22

#### Cleaning up the data
- [Link](http://ocefpaf.github.io/python4oceanographers/blog/2013/05/20/spikes/)

#### 1st Attempt

Linear Regression with Gradient Descent
- 10 features (PM2.5 of the previous 9 hours and the bias)
- Parameters
    - Learning Rate $\eta$: `1e-5`
    - Iteration Time: `1e6`
- Optimizer: None
- Score: `5.90261`

### 2/23

#### 1st Attempt

Linear Regression with Gradient Descent
- 163 features (all values of the previous 9 hours and the bias)
- Parameters
- Optimizer: None
- Score: `38.53675`

### 2/24

#### 1st Attempt

Linear Regression with Gradient Descent
- 163 features (all values of the previous 9 hours and the bias)
- Parameters
    - Learning Rate $\eta$: `1e-6`
    - Iteration Time: `1e4`
- Optimizer: None
- Score: `24.25025`

#### 2nd Attempt

Linear Regression with Gradient Descent
- 19 features (PM2.5 and PM10 of the previous 9 hours and the bias)
- Parameters
    - Learning Rate $\eta$: `1e-6`
    - Iteration Time: `1e5`
- Optimizer: None
- Score: `6.05401`

#### 3rd Attempt

Linear Regression with Gradient Descent
- 19 features (PM2.5 and PM10 of the previous 9 hours and the bias)
- Parameters
    - Learning Rate $\eta$: `1e-5`
    - Iteration Time: `5e5`
- Optimizer: None
- Score: `5.95872`

#### 4th Attempt

Linear Regression with Gradient Descent
- 19 features (PM2.5 and PM10 of the previous 9 hours and the bias)
- Parameters
    - Learning Rate $\eta$: `1e-4`
    - Iteration Time: `1e5`
- Optimizer: None
- Score: `5.90261`

#### 5th Attempt

Linear Regression with Gradient Descent
- 163 features (all values of the previous 9 hours and the bias)
- Parameters
    - Learning Rate $\eta$: `1e-6`
    - Iteration Time: `1e4`
- Optimizer: None
- Score: `5.79359`

### 2/25

#### 1st Attempt

Linear Regression with Gradient Descent
- 163 features (all values of the previous 9 hours and the bias)
- Parameters
    - Learning Rate $\eta$: `1e-6`
    - Iteration Time: `1e5`
- Optimizer: None
- Score: `5.79359`

#### 2nd Attempt

Linear Regression with Gradient Descent
- 163 features (all values of the previous 9 hours and the bias)
- Parameters
    - Learning Rate $\eta$: `1e-3`
    - Iteration Time: `1e4`
- Optimizer:
    - Feature Scaling (Standardization)
- Score: `5.79359`

#### 3rd Attempt

Linear Regression with Gradient Descent
- 163 features (all values of the previous 9 hours and the bias)
- Parameters
    - Learning Rate $\eta$: `1e-2`
    - Iteration Time: `1e4`
- Optimizer:
    - Feature Scaling (Standardization)
- Score: `5.66257`

#### ?th Attempt

Tried squaring the 9 data(PM2.5 of the previous 9 hours), Ein was large and I quitted QaQ

#### 4th Attempt

Linear Regression with Gradient Descent
- 10 features (PM2.5 of the previous 9 hours and the bias)
- Parameters
    - Learning Rate $\eta$: `1e-2`
    - Iteration Time: `1e4`
- Optimizer:
    - Feature Scaling (Standardization)
- Score: `5.90399`



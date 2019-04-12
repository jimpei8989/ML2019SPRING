# Machine Learning 2019 Spring - HW3

###### By: Wu-Jun Pei(B06902029)

### Homework Website

-   [Homework Website](https://ntumlta2019.github.io/ml-web-hw3/)

### Kaggle

-   [Kaggle Website](https://www.kaggle.com/c/ml2019spring-hw3)

|       | Public  | Private |
| ----- | ------- | ------- |
| Rank  | 40(36)  | 45(41)  |
| Score | 0.70771 | 0.69824 |

### Reproduce

-   Training Step (The result may not be exactly the same due to the GPU)

    ```bash
    bash hw3_train.sh TRAIN_CSV PATH_TO_MODEL PATH_TO_HISTORY
    ```

-   Testing Step

    ```bash
    bash hw3_test.sh TEST_CSV PREDICT_CSV
    ```

### My CNN Model

    ```text
    Input(48 * 48),
    Conv2D(128, (5, 5)), Conv2D(128, (5, 5)), BatchNormalization, MaxPooling(pool_size = (2, 2)), LeakyReLU(0.3), Dropout(0.1),
    Conv2D(256, (5, 5)), Conv2D(256, (5, 5)), BatchNormalization, MaxPooling(pool_size = (2, 2)), LeakyReLU(0.3), Dropout(0.2),
    Conv2D(512, (3, 3)), Conv2D(512, (3, 3)), BatchNormalization, MaxPooling(pool_size = (2, 2)), LeakyReLU(0.3), Dropout(0.3),
    Conv2D(768, (3, 3)), Conv2D(768, (3, 3)), BatchNormalization, MaxPooling(pool_size = (2, 2)), LeakyReLU(0.3), Dropout(0.4),
    Flatten(),
    Dense(1024), BatchNormalization, ReLU, Dropout(0.5),
    Dense(1024), BatchNormalization, ReLU, Dropout(0.5),
    Dense(512), BatchNormalization, ReLU, Dropout(0.5),
    Dense(512), BatchNormalization, ReLU, Dropout(0.5),
    Dense(7, activation = 'softmax')
    ```

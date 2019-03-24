## Machine Learning 2019 Spring - HW2 Report

<h6 style="text-align: right">學號：B06902029	系級：資工二	姓名：裴梧鈞</h6>

1.  **請比較你實作的generative model、logistic regression 的準確率，何者較佳？**

    | Model               | Public Score | Private Score |
    | ------------------- | ------------ | ------------- |
    | Generative          | 0.84004      | 0.83540       |
    | Logistic Regression | 0.85233      | 0.85136       |

    Logistic Regression 有些微較高的正確率。

2.  **請說明你實作的best model，其訓練方式和準確率為何？**

    | Model               | Public Score | Private Score |
    | ------------------- | ------------ | ------------- |
    | Gradient Boosting   | 0.87641      | 0.87483       |

    我使用的 Model 是 Gradient Boosting

    1.  在處理資料時，我有做 feature normalization
    2.  在連續的 feature，像是 *"age", "capital_gain", "capital_loss", "hours_per_week"*，我有加入二次及三次項
    3.  我做 Gradient Boosting 的參數是
        1.  `n_estimators`：173
        2.  `learning_rate`：0.05
        3.  `max_depth`：6
        4.  `random state`：將我的名字 "Wu-Jun Pei" 做 sha256sum 轉成整數模 $2^{32}$

    在選擇參數時，我有枚舉這些參數，並使用 `cross_val_score` 綜合選擇出最好的！

3.  **請實作輸入特徵標準化(feature normalization)並討論其對於你的模型準確率的影響。**

    我在前三的實作（包含 Gradient Boosting、Logistic Regression、Generative Model）都是有實作 feature normalization 的。在此，我使用的是 Logistic Regression 的 model，在沒有調整任合參數（如 learning rate、optimizer 等）的情形下，比較 feature normalization 的影響。
    | Feature Normalization | Public Score | Private Score |
    | --------------------- | ------------ | ------------- |
    | True                  | 0.85233      | 0.85136       |
    | False                 | 0.85245      | 0.85149       |

    可以看到 Feature Normalization 前的 Score 反而略高一些，我認為一種可能的原因是

    1.  資料有相當多因為 one-hot encoding 而使用 0/1 作為 feature，可以觀察到僅有 4 種 feature 是連續的，他們的影響可能沒有很大
    2.  我的 epochs 次數夠多，導致沒有 normalization 的 model 也走到一個不錯的最低點

4.  **請實作 logistic regression 的正規化 (regularization)，並討論其對於你的模型準確率的影響。**

    

5.  **請討論你認為哪個 attribute 對結果影響最大？**
# Data-Mining
[Coursework] Data Mining  

HW1 - Perceptron,BPNN Classification and BPNN Forecast   
Classification Using UCI iris data as example,Forecast Using UCI Electrical Grid as example   

HW2 - Clustering Algorithm : Self-Organizing Map (SOM)  
Clustering Using UCI iris data as example,and compare the results with MiniSOM and K-means.  

Final Project - Imbalanced data classification of semiconductor process data.  
Classification Using UCI SOCOM data ,and compare the results with other algorithm.  
半導體製程之產出預測  
半導體製造過程通常通過從感測器或過程測量點收集的信號/變數進行監控，根據這些變數預測產品的成功或失敗，具體作法如以下幾個階段：  
1.資料視覺化  
2.資料前處理：遺失值插補、離群值處理、資料正規化  
3.特徵工程：處理大量特徵的資料集，如高度共線性問題的處理、使用XGBoost進行特徵篩選、無用變數的辨別及處理  
4.資料過採樣：使用Boardline SMOTE處理高度不平衡資料集  
5.機器學習模型預測：使用Random Forest、SVM、Logistics Regression等模型進行預測，並使用Grid search調整模型參數  
6.最後使用分層交叉驗證的方式驗證預測結果，準確率達到90%、敏感度(Sensitivity)達到94%  

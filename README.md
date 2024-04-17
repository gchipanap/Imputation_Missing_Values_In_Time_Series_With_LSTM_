# Imputation Missing Values in Time Series with LSTM

## Introduction 

The rapid increase in global energy consumption has raised significant concerns regarding supply issues, depletion of energy resources, and severe environmental impacts, indicating an increase in energy consumption demand. This necessitates better management of energy resources. To effectively address these challenges, it is imperative to accurately estimate electric load demand, which requires an analysis of energy consumption patterns.

Historical energy consumption data contains observations of multiple variables over time. These patterns may be influenced by factors such as time of day, seasons, special events (holidays, workdays, etc.), and weather conditions. However, this data may contain missing values due to various reasons such as sensor failures, transmission issues, or measurement errors. These gaps in time series data can compromise the quality of analyses and forecasts.

Imputing missing values in time series data poses a significant challenge, for which various methods exist. Each method has its pros and cons. Some basic models may provide quick and easy results but at the expense of less reliable prediction. On the other hand, more complex approaches offer better results but require higher computational resources and longer execution time. In time series, recurrent models can capture temporal patterns and dependencies over time. When applied to imputation, the aim is to preserve the temporal structure and consider relationships between variables.

In this thesis, we address the challenge of imputing missing values in time series by developing and validating a model based on Long Short-Term Memory (LSTM) Neural Networks. To evaluate the model's capability, we will artificially insert 50% empty values into a complete dataset. Subsequently, the LSTM model will be trained using this modified dataset with the aim of imputing the data by predicting the missing values. The effectiveness of the model will be evaluated by comparing the results obtained with those generated using basic imputation techniques, such as forward filling, with particular emphasis on the accuracy of the time series prediction for the imputed values.

## Imputation Framework Architecture
  The architecture of the imputation framework by training an LSTM from Figure 4.2 is inspired by [3], where a time series X with a length T composed of observations x1, x2, x3..., xt is considered. The architecture comprises an LSTM layer, which during the training process in an environment without missing values, learns to capture temporal relationships and patterns in the data, and another layer called Residual Sum Vector (RSV), which captures historical information of the hidden states h1, h2, ..., ht in LSTM. A transformation matrix wimp is used to approximate the next input value based on the RSV. When a missing value is encountered in the time series, the model utilizes the RSV to estimate the missing value. During the imputation process, the model calculates the next hidden state based on the RSV and uses it to approximate the next value in the time series. If the next value is present in the data (i.e., not a missing value), the model is trained to approximate this value as part of the weight adjustment process during training. If the next value is a missing value in the data, the approximate value is directly copied as the imputation. In this case, the model does not need to be trained to approximate the missing value, as the calculated approximation using the hidden state and other model parameters is simply utilized. During training, a loss function consisting of two main components is optimized: the approximation loss and the task-related loss. The approximation loss is calculated as the mean squared error between the original values and the imputed values, considering the presence or absence of the original values.

The loss function used during model training consists of two main components: the approximation loss and the task-related loss. The approximation loss is calculated as the mean squared error between the original values and the imputed values, considering the presence or absence of the original values. The task-related loss is associated with the specific goal of the model, such as classification or regression. Mathematically, the RSV at time t, denoted as rt, is defined as:
rt =
(
p(ht), if t = 1
p(ht + q(Wr, rt−1)), if t = 2, 3, ..., T
)

Where p and q are vector functions, Wr is a transformation matrix, ht is the output of the hidden layer of LSTM at time t, and rt−1 is the RSV at the previous time. The approximation of the next value in the time series, zt+1, is calculated using the RSV:
zt+1 = Wimp · rt
During training, a total loss function, Ltotal, is optimized, which combines the approximation loss and the task-related loss. The total loss is defined as:

Ltotal = Ltotal_{approx} + λ_{target} \cdot Ltotal_{target}

 
Where Ltotal_{\text{approx}} is the total approximation loss, Ltotal_{\text{target}} is the total task-related loss, and λ_{\text{target}} is a coefficient that weights the importance of the task-related loss. This loss function is optimized using the Back Propagation Through Time (BPTT) algorithm.
## Experimentation and Results
### Hyperparameters
learning_rate = 0.01
batch_size = 4
hidden_size = 128
epochs = 150
evaluation_metric = MSE
### Dataset
The dataset used is related to the energy consumption of three different distribution networks in the city of Tetouan, located in northern Morocco. The energy consumption measured in these 3 networks spans the 12 months of the year 2017 with measurements taken every 10 minutes.
#### Preprocessing
For data preprocessing, the MinMaxScaler scaler is used to compute and store the minimum and maximum values of each consumption zone in the dataset.
#### Generating empty values
Since the data is complete, missing values are artificially generated in 10%, 30%, 50%, 70%, and 90% of the data. These missing values are masked with a value of -1.0 to be able to identify them and not affect the model with NaN values. Table 4.2 shows the distribution of the dataset with the generated missing values.
#### Preprocessing 2
Since this dataset contains missing values masked with -1.0, first these -1.0 values are replaced with NaN. Subsequently, the normalizer previously fitted to the complete dataset is applied. Finally, the NaN values are replaced again with -1.0.




# Tesla Stock Price Prediction


## Project Overview
This project focuses on analyzing and forecasting Tesla's stock prices using a combination of traditional statistical models and deep learning techniques. The primary goal is to predict the future behavior of Tesla's stock, leveraging data from 2010 to 2022.

## Objectives
1. Perform Exploratory Data Analysis (EDA) to understand stock price trends.
2. Check data stationarity and identify seasonal components.
3. Develop predictive models:
   - SARIMAX: A seasonal autoregressive integrated moving average model with exogenous factors.
   - LSTM: A deep learning-based Long Short-Term Memory model for time series forecasting.
4. Evaluate models using RMSE (Root Mean Squared Error).

## Dataset
The dataset consists of Tesla stock price data obtained from a public source. Key columns include:
- `Date`: Trading date.
- `Open`, `High`, `Low`, `Close`: Stock prices at different times of the trading day.
- `Adj Close`: Adjusted closing price.
- `Volume`: Number of shares traded.

### Data Preprocessing
- Missing values were checked and handled.
- Date column was parsed and set as the index.
- Rolling averages (50-day and 200-day) were calculated to capture trends.

## Methodology

### Exploratory Data Analysis (EDA)
- Visualized stock price trends over time.
- Computed summary statistics and moving averages.
- Conducted the Augmented Dickey-Fuller (ADF) test to check for stationarity.
- Performed seasonal decomposition to analyze trends and seasonality.

### Modeling Approaches
#### SARIMAX
- Captures both seasonality and external influences on stock prices.
- Optimal parameters selected using AIC (Akaike Information Criterion).
- Predictions and confidence intervals plotted against actual prices.

#### LSTM
- Scaled data to a 0-1 range using MinMaxScaler.
- Prepared sequences of historical data to predict the next price.
- Built a stacked LSTM model using Keras:
  - Two LSTM layers with 50 neurons each.
  - Dense layers for output prediction.
- Trained the model on a subset of data and evaluated on the test set.

## Results
- **SARIMAX Model**:
  - Effectively captured seasonality and trends.
  - RMSE: Evaluated on test data for prediction accuracy.
- **LSTM Model**:
  - Leveraged deep learning for time series forecasting.
  - RMSE: Showed comparable performance with SARIMAX.

## Conclusion
- The SARIMAX model provided robust results with interpretable components.
- The LSTM model offered an alternative, leveraging historical data for predictive insights.
- Both models have strengths; a hybrid approach could improve accuracy further.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/RenaAbbasova/Tesla_stock_prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset (`TSLA.csv`) in the project directory.
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Tesla_Stock_Prediction.ipynb
   ```

## Dependencies
- Python 3.7+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Keras
- TensorFlow
- Scikit-learn

## Files
- `Tesla_Stock_Prediction.ipynb`: Jupyter Notebook with analysis, modeling, and conclusions.
- `TSLA.csv`: Dataset with Tesla stock prices.
- `README.md`: Project documentation.

## Future Work
- Incorporate additional exogenous variables (e.g., market indices, news sentiment).
- Explore other deep learning architectures like GRU or Transformer models.
- Develop an ensemble model combining SARIMAX and LSTM predictions.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
- Tesla stock data source: Yahoo Finance.
- Inspiration: Time series forecasting applications in finance.

![Project Illustration](tesla.png)


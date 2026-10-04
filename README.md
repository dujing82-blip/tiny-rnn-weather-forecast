# Tiny RNN Weather Forecast 🌦️

A beginner-friendly TensorFlow project for learning how a **Recurrent Neural Network (RNN)** can predict the next value in a time sequence.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/tiny-rnn-weather-forecast/blob/main/RNN_Weather_Forecast.ipynb)

## The whole idea

```text
last 14 days of temperature
            ↓
        SimpleRNN
            ↓
 tomorrow's temperature
```

The project includes a ready-to-use CSV dataset with three years (1,095 days) of daily temperature. Students download the data from GitHub, visualize it, turn it into sequences, and train an RNN.

## What students learn

- What a **sequence** is
- Why order matters in time-series data
- How to make sliding 14-day input windows
- The input shape of an RNN: `[examples, time steps, features]`
- What the RNN hidden state / memory means
- How `SimpleRNN` differs conceptually from a Dense network
- Why temperature prediction is a regression problem
- How to compare predictions with real values

## Model

```text
14 days × 1 temperature
          ↓
     SimpleRNN(32)
          ↓
       Dense(1)
          ↓
next-day temperature
```

The code deliberately uses **SimpleRNN**, not LSTM or GRU, so students can first understand the basic recurrent idea.

## Data

The repository includes `weather_temperature.csv`, containing 1,095 daily temperature observations with realistic seasonal and short-term variation.

The notebook loads it directly with Pandas:

```python
data = pd.read_csv(DATA_URL)
```

No data-generation code is needed in the lesson. The first two years are used for training and the third year is kept for testing.

## Run

Click **Open in Colab** above and choose **Runtime → Run all**.

## Requirements

Everything is available in Google Colab:

- TensorFlow / Keras
- NumPy
- Matplotlib

## License

MIT License

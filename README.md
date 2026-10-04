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

The notebook creates three years of simulated daily temperature. The data contains a clear yearly seasonal cycle, multi-day warm/cold weather changes, and small daily noise.

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

No external dataset or upload is required. The notebook creates its own weather:

```python
season = 65 + 20 * np.sin(2 * np.pi * (days - 80) / 365)
```

A smooth yearly seasonal pattern is combined with short-term weather variation and daily noise.

The first two years are used for training. The third year is kept for testing.

## Run

Click **Open in Colab** above and choose **Runtime → Run all**.

## Requirements

Everything is available in Google Colab:

- TensorFlow / Keras
- NumPy
- Matplotlib

## License

MIT License

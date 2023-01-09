

1. Make sure the buffer never contains samples that
not from a past task (monitor the data_valuation part)
2. Should give more priority to training on a new task.
Local training:
    Loop through data in this task
        sample from buffer from old tasks
        concat data
        train

Sharing
Data Valuation & receive
    each shared data task is already in the buffer and the data valuation is overwhelmingly one-sided.
        - add all data to buffer
        - solve by backtesting again...

Running into OOM issue with this dataset...


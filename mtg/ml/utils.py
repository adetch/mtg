import numpy as np
import tensorflow as tf
import pickle
import os


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    learning rate scheduling
    """

    def __init__(self, d_model, warmup_steps=1000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def importance_weighting(df, minim=0.1, maxim=1.0):
    # Player skill (rank / run-wins) and format are now CONDITIONING INPUTS to the
    # model rather than sample weights, and the PxP1 "reduce rare drafting"
    # position down-weight has been dropped (position remains a model feature, just
    # not a weight). So sample weighting keeps ONLY recency: it is about metagame
    # drift over time, not about pick quality. See state/sos_conditioning_plan.md.
    last = df["date"].max()
    # increase importance factor for recent data points according to number of weeks from most recent data point
    n_weeks = df["date"].apply(lambda x: (last - x).days // 7)
    return 0.9 ** n_weeks


def load_model(location, extra_pickle="attrs.pkl"):
    model_loc = os.path.join(location, "model")
    data_loc = os.path.join(location, extra_pickle)
    model = tf.saved_model.load(model_loc)
    try:
        with open(data_loc, "rb") as f:
            extra = pickle.load(f)
        return (model, extra)
    except:
        return model

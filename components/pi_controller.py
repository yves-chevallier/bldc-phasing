import numpy as np

class PIController:
    def __init__(self, kp, ki, dt, output_limits=(-np.inf, np.inf)):
        self.kp = kp
        self.ki = ki
        self.dt = dt
        self.integral = 0
        self.min_out, self.max_out = output_limits

    def reset(self):
        self.integral = 0

    def compute(self, error):
        self.integral += error * self.dt
        output = self.kp * error + self.ki * self.integral
        return np.clip(output, self.min_out, self.max_out)

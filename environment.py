import gymnasium as gym
import serial
import time
import numpy as np

class CartPoleEnv:
    def __init__(self, mode="sim", serial_port=None, baudrate=115200):
        self.mode = mode
        if mode == "sim":
            self.env = gym.make("CartPole-v1")
        elif mode == "real":
            assert serial_port is not None, "Serial port muss angegeben werden!"
            self.serial = serial.Serial(serial_port, baudrate=baudrate, timeout=1)
            time.sleep(2)  # Serial initialisieren lassen

    def reset(self):
        if self.mode == "sim":
            obs, info = self.env.reset()
            #if isinstance(obs, tuple):
            #    obs = obs[0]
            return obs, info
        elif self.mode == "real":
            #self.serial.write(b'RESET\n')
            input("CartPole bitte wieder in die Startposition bringen und Enter drücken...")
            line = self.serial.readline().decode().strip()
            obs = np.array([float(x) for x in line.split(',')])
            return obs, {}

    def step(self, action):
        if self.mode == "sim":
            # Policy gibt float, CartPole erwartet int 0/1
            discrete_action = int(action >= 0)
            obs, reward, terminated, truncated, info = self.env.step(discrete_action)
            return obs, reward, terminated, truncated, info

        elif self.mode == "real":
            action_str = f"{float(action)}\n"  # TODO: Action normalisieren
            self.serial.write(action_str.encode())
            line = self.serial.readline().decode().strip()
            # Erwartet: obs als CSV, reward, done als 0/1, ggf. info
            parts = line.split(' ')
            # TODO: state berechnen
            obs = np.array([float(x) for x in parts[0].split(',')])
            reward = float(parts[1])
            terminated = bool(int(parts[2]))
            truncated = bool(int(parts[3]))
            info = {}
            return obs, reward, terminated, truncated, info

    def close(self):
        if self.mode == "sim":
            self.env.close()
        elif self.mode == "real":
            self.serial.close()

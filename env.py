from controller import Robot, Supervisor, PositionSensor
import numpy as np


class Env:

    def __init__(self, target=(0, 5, 0)):

        self.robot = Supervisor()

        self.MotorArgs = []
        self.MotorArgs.append(self.robot.getDevice('leg1_motor1'))
        self.MotorArgs.append(self.robot.getDevice('leg1_motor2'))
        self.MotorArgs.append(self.robot.getDevice('leg1_motor3'))
        self.MotorArgs.append(self.robot.getDevice('leg2_motor1'))
        self.MotorArgs.append(self.robot.getDevice('leg2_motor2'))
        self.MotorArgs.append(self.robot.getDevice('leg2_motor3'))
        self.MotorArgs.append(self.robot.getDevice('leg3_motor1'))
        self.MotorArgs.append(self.robot.getDevice('leg3_motor2'))
        self.MotorArgs.append(self.robot.getDevice('leg3_motor3'))
        self.MotorArgs.append(self.robot.getDevice('leg4_motor1'))
        self.MotorArgs.append(self.robot.getDevice('leg4_motor2'))
        self.MotorArgs.append(self.robot.getDevice('leg4_motor3'))
        self.MotorArgs.append(self.robot.getDevice('leg5_motor1'))
        self.MotorArgs.append(self.robot.getDevice('leg5_motor2'))
        self.MotorArgs.append(self.robot.getDevice('leg5_motor3'))
        self.MotorArgs.append(self.robot.getDevice('leg6_motor1'))
        self.MotorArgs.append(self.robot.getDevice('leg6_motor2'))
        self.MotorArgs.append(self.robot.getDevice('leg6_motor3'))

        self.ps = []
        self.ps.append(PositionSensor('position_sensor_leg1_motor1'))
        self.ps.append(PositionSensor('position_sensor_leg1_motor2'))
        self.ps.append(PositionSensor('position_sensor_leg1_motor3'))
        self.ps.append(PositionSensor('position_sensor_leg2_motor1'))
        self.ps.append(PositionSensor('position_sensor_leg2_motor2'))
        self.ps.append(PositionSensor('position_sensor_leg2_motor3'))
        self.ps.append(PositionSensor('position_sensor_leg3_motor1'))
        self.ps.append(PositionSensor('position_sensor_leg3_motor2'))
        self.ps.append(PositionSensor('position_sensor_leg3_motor3'))
        self.ps.append(PositionSensor('position_sensor_leg4_motor1'))
        self.ps.append(PositionSensor('position_sensor_leg4_motor2'))
        self.ps.append(PositionSensor('position_sensor_leg4_motor3'))
        self.ps.append(PositionSensor('position_sensor_leg5_motor1'))
        self.ps.append(PositionSensor('position_sensor_leg5_motor2'))
        self.ps.append(PositionSensor('position_sensor_leg5_motor3'))
        self.ps.append(PositionSensor('position_sensor_leg6_motor1'))
        self.ps.append(PositionSensor('position_sensor_leg6_motor2'))
        self.ps.append(PositionSensor('position_sensor_leg6_motor3'))

        self.MotorPosition = np.zeros((1, 18), dtype=np.float32)

        self.observation_space = np.zeros((1, 18), dtype=np.float32)
        self.action_space = np.zeros((1, 18), dtype=np.float32)

        self.timestep = int(self.robot.getBasicTimeStep())

        for i in range(18):
            self.ps[i].enable(self.timestep)

        self.target = target

    def step(self, action):
        # target is a list with 3 floats
        self.robot.step(self.timestep)

        # execute action
        self.MotorPosition += action

        # update observation
        for i in range(18):
            self.MotorArgs[i].set_position(self.MotorPosition[i])
            self.MotorPosition[i] = self.ps[i].getValues()

        # calculate rewards, use supervisor to get robot position
        robot_node = self.robot.getSelf()

        trans_field = robot_node.getField("translation")

        position = trans_field.getSFVec3f()

        dis = np.sum((position - self.target) * (position - self.target))
        reward = 1 / dis ** (1 / 2)
        # avoid breaking down when the robot is close to the target
        if dis < 10000:
            d = False
        else:
            d = True

        return self.MotorPosition, reward, d, False

    def reset(self):
        self.robot.simulationReset()

        self.robot.step(self.timestep)

        self.MotorPosition = np.zeros((1, 18), dtype=np.float32)

        return self.MotorPosition

    def reward(self, position):
        dis = np.sum((position - self.target) * (position - self.target))
        reward = 1 / dis ** (1 / 2)
        return reward

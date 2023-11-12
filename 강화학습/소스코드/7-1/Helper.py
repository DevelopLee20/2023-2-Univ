import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 50  # 픽셀 수
HEIGHT = 5  # 그리드 세로
WIDTH = 5  # 그리드 가로

np.random.seed(1)


class Env(tk.Tk):
    def __init__(self, name, render_speed=0.01):
        super(Env, self).__init__()
        self.render_speed=render_speed
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title(name)
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.counter = 0
        self.rewards = []  # 장애물, 목표지점을 나타내는 딕셔너리의 리스트
                           # 딕셔너리의 키: reward, state, direction, coords, figure
        self.goal = []
        # 장애물 설정
        self.set_reward([0, 1], -1)
        self.set_reward([1, 2], -1)
        self.set_reward([2, 3], -1)
        # 목표 지점 설정
        self.set_reward([4, 4], 1)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # 그리드 생성
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        self.rewards = []
        self.goal = []
        # 캔버스에 이미지 추가
        x, y = UNIT/2, UNIT/2
        self.rectangle = canvas.create_image(x, y, image=self.shapes[0])

        canvas.pack()                

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("./img/rectangle.png").resize((30, 30)))
        triangle = PhotoImage(
            Image.open("./img/triangle.png").resize((30, 30)))
        circle = PhotoImage(
            Image.open("./img/circle.png").resize((30, 30)))

        return rectangle, triangle, circle

    def reset_reward(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()
        self.set_reward([0, 1], -1)
        self.set_reward([1, 2], -1)
        self.set_reward([2, 3], -1)

        # #goal
        self.set_reward([4, 4], 1)

    # 각 상태에 대해 reward 딕셔너리 생성
    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward > 0:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                       (UNIT * y) + UNIT / 2,
                                                       image=self.shapes[2])

            self.goal.append(temp['figure'])


        elif reward < 0:
            temp['direction'] = -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[1])

        temp['coords'] = self.canvas.coords(temp['figure'])
        temp['state'] = state
        self.rewards.append(temp)

    # 에이전트가 장애물, 목표지점 도달 시 체크 리스트(딕셔너리) 리턴
    def check_if_reward(self, state):
        check_list = dict()
        check_list['if_goal'] = False
        rewards = 0

        for reward in self.rewards:
            if reward['state'] == state:
                rewards += reward['reward']       # 장애물 충돌 시 마다 보상 -1 누적
                if reward['reward'] == 1:
                    check_list['if_goal'] = True

        check_list['rewards'] = rewards

        return check_list

    def coords_to_state(self, coords):
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        self.reset_reward()
        return self.get_state()

    # 한 스텝 진행
    def step(self, action):
        self.counter += 1
        self.render()
        
        # 2 스텝에 한 번씩 장애물 이동
        if self.counter % 2 == 1:
            self.rewards = self.move_rewards()    # 장애물 이동 후의 reward 계산

        # 에이전트 이동
        next_coords = self.move(self.rectangle, action)
        # 장애물 충돌 체크
        check = self.check_if_reward(self.coords_to_state(next_coords))
        done = check['if_goal']      # 목표지점 도달 시
        reward = check['rewards']    # 중복 장애물 충돌 시누적 보상

        self.canvas.tag_raise(self.rectangle)

        s_ = self.get_state()    # 에이전트의 도착지점, 장애물에 대한 상대 위치, 라벨, 이동 방향

        return s_, reward, done

    # 에이전트 이동 후 상태(상대 위치, 라벨) 등
    def get_state(self):

        location = self.coords_to_state(self.canvas.coords(self.rectangle))
        agent_x = location[0]
        agent_y = location[1]

        states = list()

        for reward in self.rewards:
            reward_location = reward['state']
            states.append(reward_location[0] - agent_x)
            states.append(reward_location[1] - agent_y)
            # 장애물이면
            if reward['reward'] < 0:
                states.append(-1)    # 장애물 표시 라벨
                states.append(reward['direction'])   # 장애물의 이동 방향
            else:
                states.append(1)   # 목표 지점 라벨

        return states

    # 장애물 이동, 이동 후 변경된 위치의 reward 딕셔너리 리스트 리턴
    def move_rewards(self):        
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] == 1:       # 목표 지점이면
                new_rewards.append(temp)
                continue
            temp['coords'] = self.move_const(temp)   # 이동 후 좌표
            temp['state'] = self.coords_to_state(temp['coords'])  # 이동 후 위치
            new_rewards.append(temp)
        return new_rewards

    # 이동 후 좌표 리턴
    def move_const(self, target):

        s = self.canvas.coords(target['figure'])

        base_action = np.array([0, 0])

        # 오른 쪽 끝이면 
        if s[0] == (WIDTH - 1) * UNIT + UNIT / 2:
            target['direction'] = 1      # 왼쪽 방향 이동
        # 왼쪽 끝이면
        elif s[0] == UNIT / 2:          # 오른쪽 이동
            target['direction'] = -1

        # 오른쪽 이동이면 픽셀 좌표 증가
        if target['direction'] == -1:
            base_action[0] += UNIT
        # 왼쪽 이동이면 픽셀 좌표 감소
        elif target['direction'] == 1:
            base_action[0] -= UNIT
        # ? rectangle
        if (target['figure'] is not self.rectangle
           and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]):
            base_action = np.array([0, 0])

        self.canvas.move(target['figure'], base_action[0], base_action[1])

        s_ = self.canvas.coords(target['figure'])

        return s_

    def move(self, target, action):
        s = self.canvas.coords(target)

        base_action = np.array([0, 0])

        if action == 0:  # 상
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 하
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 우
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # 좌
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(target, base_action[0], base_action[1])

        s_ = self.canvas.coords(target)

        return s_

    def render(self):
        # 게임 속도 조정
        time.sleep(self.render_speed)
        self.update()
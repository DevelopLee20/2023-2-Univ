import numpy as np
from tqdm import *
from collections import defaultdict
from GridSQ_environment import Env, WIDTH, HEIGHT
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def sarsa(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, 
          alpha_decay_ratio=0.5, init_epsilon=1.0, min_epsilon=0.1, 
          epsilon_decay_ratio=0.9, n_episodes=3000):
    nS = WIDTH * HEIGHT
    nA = 4
    alpha = init_alpha
    epsilon = init_epsilon
    
    Q = np.zeros((nS, nA), dtype=np.float64)
    q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    select_action = lambda state, Q, eps: \
        np.argmax(Q[state]) if np.random.random() > eps else np.random.randint(len(Q[state]))

    exit_track = []
    
    for e in tqdm(range(n_episodes), desc="All Episodes"):
        init_xy, done = env.reset(), False
        state = state_num(init_xy)
        action = select_action(state, Q, epsilon)

        while not done:
            next_xy, reward, done = env.step(action)
            next_state = state_num(next_xy)
            next_action = select_action(next_state, Q, epsilon)
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]

            # alphas에 대한 정의가 없다. (alphas는 alpha의 경험 튜플일 것이다.)
            # Q[state][action] = Q[state][action] + alphas[e] * td_error
            Q[state][action] = Q[state][action] + alpha * td_error

            x = state % WIDTH
            y = state // WIDTH
            state_xy = [x, y]
            q_table[str(state_xy)][action] = Q[state][action]
            state, action = next_state, next_action
            env.print_value_all(q_table)
        
        # 하나의 에피소드가 종료할 때 마다 알파값과 입실론값을 감가
        if (alpha > min_alpha):
            alpha *= alpha_decay_ratio

        if (epsilon > min_epsilon):
            epsilon *= epsilon_decay_ratio

        # 도착 후 가치가 100을 달성하면, 성공으로 판단
        if (reward == 100):
            exit_track.append(1)
        else:
            exit_track.append(0)

    goal_num = exit_track.count(1)
    goal_rate = goal_num / len(exit_track)
    fail_num = exit_track.count(0)
    fail_rate = fail_num / len(exit_track)
    print(f"목표지점 도달 {goal_num:3d} {goal_rate:5.2f}%, 장애물 도달 {fail_num:3d} {fail_rate:5.2f}%")

    return exit_track

# Q-학습(Q-Learning) 알고리즘
def q_learning(env, gamma=1.0,
               init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS = WIDTH * HEIGHT # 5x5 격자
    nA = 4              
    alpha = init_alpha
    epsilon = init_epsilon
    
    Q = np.zeros((nS, nA), dtype=np.float64)
    q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    select_action = lambda state, Q, eps: \
        np.argmax(Q[state]) if np.random.random() > eps else np.random.randint(len(Q[state]))

    exit_track = []
    
    for e in tqdm(range(n_episodes), desc="All Episodes"):
        init_xy, done = env.reset(), False
        state = state_num(init_xy)
        action = select_action(state, Q, epsilon)

        while not done:
            next_xy, reward, done = env.step(action)
            next_state = state_num(next_xy)
            next_action = select_action(next_state, Q, epsilon)
            td_target = reward + gamma * Q[next_state].max() * (not done) # SarSa와 다른 코드
            td_error = td_target - Q[state][action]

            Q[state][action] = Q[state][action] + alpha * td_error
            
            x = state % WIDTH
            y = state // WIDTH
            state_xy = [x, y]
            q_table[str(state_xy)][action] = Q[state][action]
            state, action = next_state, next_action
            env.print_value_all(q_table)
        
        # 하나의 에피소드가 종료할 때 마다 알파값과 입실론값을 감가
        if (alpha > min_alpha):
            alpha *= alpha_decay_ratio

        if (epsilon > min_epsilon):
            epsilon *= epsilon_decay_ratio

        # 도착 후 가치가 100을 달성하면, 성공으로 판단
        if (reward == 100):
            exit_track.append(1)
        else:
            exit_track.append(0)

    goal_num = exit_track.count(1)
    goal_rate = goal_num / len(exit_track)
    fail_num = exit_track.count(0)
    fail_rate = fail_num / len(exit_track)
    print(f"목표지점 도달 {goal_num:3d} {goal_rate:5.2f}%, 장애물 도달 {fail_num:3d} {fail_rate:5.2f}%")

    return exit_track

plt.style.use("fivethirtyeight")
params = {
    "figure.figsize": (15, 8),
    "font.size": 24,
    "legend.fontsize": 20,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
}
pylab.rcParams.update(params)

if __name__ == "__main__":
    # 환경 설정
    env = Env("Sarsa")
    state_num = lambda s: s[0] + HEIGHT * s[1]

    # 하이퍼 파라미터 설정
    gamma = 0.99
    max_episodes = 300

    done_track = sarsa(env, gamma=gamma, n_episodes=max_episodes)

    # 그래프 출력
    plt.plot(done_track)
    plt.title("Sarsa: Grid World")
    plt.xlabel("episode")
    plt.ylabel("fail, goal")
    plt.show()

import torch
import numpy as np
import time
import random
import gym
import matplotlib

# torch 라이브러리의 nn.Module를 상속받아서 클래스를 구현합니다.
# nn.Module은 모든 신경망 모듈의 기본 클래스 입니다.
class FCQ(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32), activation_fc=torch.nn.functional.relu):
        super(FCQ, self).__init__()         # 기본 클래스의 매개변수들을 그대로 사용합니다.
        self.activation_fc = activation_fc  # 활성화 함수, 기본값은 relu

        self.input_layer = torch.nn.Linear(input_dim, hidden_dims[0])   # 입력 레이어입니다.
        self.hidden_layers = torch.nn.ModuleList()  # 동적인 은닉층을 생성할 때, 리스트에 넣어 사용합니다.

        for i in range(len(hidden_dims)-1): # 은닉층 차원수만큼 생성합니다.
            hidden_layer = torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]) # 은닉층 완전연결계층을 생성합니다.
            self.hidden_layers.append(hidden_layer) # 생성된 완전연결계층을 ModuleList에 추가합니다.
        
        self.output_layer = torch.nn.Linear(hidden_dims[-1], output_dim) # 출력 레이어입니다.

        device = "cpu"                  # 기본 device cpu로 설정
        if torch.cuda.is_available():   # torch의 cuda 설정 가능할 경우
            device = "cuda:0"           # device cuda로 설정
        
        self.device = torch.device(device)  # 설정된 device로 학습
        self.to(self.device)                # 모듈의 파라미터/버퍼 등을 해당 디바이스로 이동 (casting)

    def _format(self, state):   # 입력값이 텐서가 아니면 텐서로 변환
        x = state

        if not isinstance(x, torch.Tensor): # 입력값의 인스턴스가 파이토치 텐서인지 비교
            x = torch.tensor(x, device=self.device, dtype=torch.float32)    # 아닐 경우 토치형식으로 새로 생성
            x = x.unsqueeze(0)
        
        return x
    
    def forward(self, state):   # 신경망 순전파 함수
        x = self._format(state)                     # 토치 텐서로 변환
        # x = self.activate_fc(self.input_layer(x))   # 활성화 함수 설정
        x = self.activation_fc(self.input_layer(x))   # 코드 수정

        for hidden_layer in self.hidden_layers:     # 은닉층의 수에 따라 반복
            # x = self.activate_fc(hidden_layer(x))   # 활성화 함수 지정
            x = self.activation_fc(hidden_layer(x))   # 코드 수정
        
        x = self.output_layer(x)                    # 출력 레이어 설정

        return x
    
    def load(self, experiences): # 경험 튜플을 텐서로 변환해주는 메서드
        states, actions, new_states, rewards, is_terminals = experiences

        states = torch.from_numpy(states).float().to(self.device)               # 상태
        actions = torch.from_numpy(actions).long().to(self.device)              # 행동
        new_states = torch.from_numpy(new_states).float().to(self.device)       # 다음 상태
        rewards = torch.from_numpy(rewards).float().to(self.device)             # 보상
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)   # 종료 여부

        return states, actions, new_states, rewards, is_terminals
    
class EGreedyStrategy(): # 입실론 그리디 탐색 전략으로 훈련을 진행합니다.
    def __init__(self, epsilon=0.1): # 기본 입실론 값은 0.1으로 지정
        self.epsilon = epsilon
        self.exploratory_action_taken = None # 선택한 행동이 탐색인지에 대한 여부 확인

    def select_action(self, model, state):
        self.exploratory_action_taken = False

        with torch.no_grad(): # 연산 기록을 하지 않아, 검증 속도가 향상됩니다.
            q_values = model(state).cpu().detach().data.numpy().squeeze()
        
        if np.random.rand() > self.epsilon: # 입실론 값보다 크면
            action = np.argmax(q_values)    # 최대 보상을 가진 행동으로 선택합니다.
        else:                                           # 입실론 값보다 작으면
            action = np.random.randint(len(q_values))   # 랜덤의 행동을 선택합니다.
        
        self.exploratory_action_taken = action != np.argmax(q_values) # 탐색 여부 저장

        return action
    
# 평가 시에는 최대 보상을 가진 행동을 선택하도록 한다.
class GreedyStrategy():
    def __init__(self):
        self.exploratory_action_taken = False # 탐색인지 확인
    
    def select_action(self, model, state): # 행동을 선택하는 메서드
        with torch.no_grad(): # 기울기를 저장하지 않아, 검증 속도가 향상됩니다.
            q_values = model(state).cpu().detach().data.numpy().squeeze()
        
        return np.argmax(q_values)

class NFQ():
    def __init__(self, value_model_fn, value_optimizer_fn, value_optimizer_lr, training_strategy_fn, evaluation_strategy_fn, batch_size, epochs):
        self.value_model_fn = value_model_fn                    # 가치함수
        self.value_optimizer_fn = value_optimizer_fn            # 최적화함수
        self.value_optimizer_lr = value_optimizer_lr            # 최적화함수의 학습률
        self.training_strategy_fn = training_strategy_fn        # 훈련 전략
        self.evaluation_strategy_fn = evaluation_strategy_fn    # 정책 개선 전략
        self.batch_size = batch_size                            # 배치크기
        self.epochs = epochs                                    # 학습횟수

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        # Q-학습 타겟 계산 후 다음 상태 최대 Q-함수 값의 행동 가치함수 사용하여 타겟 계산
        max_a_q_sp = self.online_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_s = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        # 상태 Q-함수 예측, gather 연산으로 인덱스의 Q-함수 값 추출
        q_sa = self.online_model(states).gather(1, actions)

        td_errors = q_sa - target_q_s                   # 손실 함수
        value_loss = td_errors.pow(2).mul(0.5).mean()   # 오차 계산
        self.value_optimizer.zero_grad()                # 최적화 함수 기울기 초기화
        value_loss.backward()                           # 역전파 발산
        self.value_optimizer.step()                     # 최적화 진행

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state) # 행동 선택
        new_state, reward, terminated, truncated, info = env.step(action)       # step 진행
        done = terminated or truncated                                          # 종료 여부 확인
        experience = (state, action, reward, new_state, float(done))            # 경험 튜플 저장
        self.experiences.append(experience)                                     # 경험 튜플 리스트 추가
        self.episode_reward[-1] += reward                                       # 보상 누적
        self.episode_timestep[-1] += 1                                          # 타임 스텝 누적
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken) # 탐색 횟수 누적
        
        return new_state, done
    
    def train(self, env, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward):
        # 학습전 초기화
        training_start, last_debug_time = time.time(), float('-inf')
        self.seed = seed
        self.gamma = gamma
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # 학습전 초기화 끝

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []
        self.online_model = self.value_model_fn(nS, nA)
        self.value_optimizer = self.value_optimizer_fn(self.online_model, self.value_optimizer_lr)
        self.training_strategy = self.training_strategy_fn() # self.이 없어서 코드 수정
        self.evaluation_strategy = self.evaluation_strategy_fn() # self.이 없어서 코드 수정
        self.experiences = []

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0

        # 에피소드 횟수 만큼 반복
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()                     # 에피소드 시작시 시간 체크
            start, done = env.reset(seed=self.seed), False  # 환경 초기화
            state = start[0]
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            # 종료 상태 도달 전까지 반복
            while not done:
                state, done = self.interaction_step(state, env) # step 진행

                # 배치사이즈 만큼 step 기록을 저장
                if len(self.experiences) >= self.batch_size:
                    experiences = np.array(self.experiences, dtype=object)
                    batches = [np.vstack(sars) for sars in experiences.T] # T는 전치연산
                    experiences = self.online_model.load(batches)

                    for _ in range(self.epochs):
                        self.optimize_model(experiences)
                    
                    self.experiences.clear()

            # 종료 상태 도달시 결과 계산 시작
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            total_step = int(np.sum(self.episode_timestep))
            evaluation_score, _ = self.evaluate(self.online_model, env)
            self.evaluation_scores.append(evaluation_score)
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)
            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, mean_100_eval_score, mean_100_exp_rat, wallclock_elapsed
            LEAVE_PRINT_EVERY_N_SECS = 60
            ERASE_LINE = "\x1b[2K"
            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_reward >= goal_mean_100_reward
            training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, total_step,
                mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward,
                mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)

            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            
            if training_is_over:
                if reached_max_minutes: print(u"--> reached_max_minutes \u2715")
                if reached_max_episodes: print(u"--> reached_max_episodes \u2715")
                if reached_goal_mean_reward: print(u"--> reached_goal_mean_reward \u2713")
                break
        
        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print("Training complete.")
        print(f"Final evaluation score {final_eval_score:.2f}\u00B1{score_std:.2f} in {training_time:.2f}s training time, {wallclock_time:.2f}s wall-clock time.\n")
        env.close()
        del env

        return result, final_eval_score, training_time, wallclock_time
        
    # 에피소드 및 훈련 종료 후 그리디 전략으로 보상을 계산하는 메소드
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = [] # 보상 리스트 초기화

        for _ in range(n_episodes):             # 에피소드 만큼 반복
            s, done = eval_env.reset(), False   # 환경 초기화
            s = s[0]                            # 상태 저장
            rs.append(0)                        # 보상 리스트에 공간 할당

            # 에피소드 종료시까지 반복
            while not done:
                a = self.evaluation_strategy.select_action(eval_policy_model, s) # 행동을 선택
                s, r, terminated, truncated, info = eval_env.step(a)             # 하나의 스텝 진행
                done = terminated or truncated                                   # 종료 여부 판단
                rs[-1] += r                                                      # 리스트 끝에 계산된 보상 저장
        
        return np.mean(rs), np.std(rs)  # 보상의 평균값과 표준값을 반환

    # 학습 후 그리디 전략으로 렌더링 해주는 메소드
    def render_after_train(self, r_env, n_episodes=1):
        for _ in range(n_episodes):         # 에피소드 횟수만큼 반복
            s, done = r_env.reset(), False  # 환경 초기화
            s = s[0]                        # 상태 저장

            # 에피소드 종료시까지 반복
            while not done:
                a = self.evaluation_strategy.select_action(self.online_model, s)    # 행동 선택
                s, r, terminated, truncated, info = r_env.step(a)                   # 스텝 진행
                done = terminated or truncated                                      # 종료 여부 반환

nfq_results = []
SEEDS = (12, 34, 56, 78, 90)

for seed in SEEDS:
    environment_settings = {
        "env_name": "CartPole-v1",
        "gamma": 1.0,
        "max_minutes": 20,
        "max_episodes": 10000,
        "goal_mean_100_reward": 475
    }

    value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512, 128))
    value_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005
    training_strategy_fn = lambda: EGreedyStrategy(epsilon=0.5)
    evaluation_strategy_fn = lambda: GreedyStrategy()
    batch_size = 1024
    epochs = 40

    env_name, gamma, max_minutes, max_episodes, goal_mean_100_reward = environment_settings.values()
    env = gym.make(env_name)
    agent = NFQ(value_model_fn, value_optimizer_fn, value_optimizer_lr, training_strategy_fn, evaluation_strategy_fn, batch_size, epochs)
    result, eval_score, training_time, wallclock_time = agent.train(env, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    nfq_results.append(result)

nfq_results = np.array(nfq_results)
# env = gym.make(env_name, render_mode="human")
env = gym.make(env_name) # Jupyter 상에서 진행함으로 코드 수정
agent.render_after_train(env)
# env.close()

nfq_max_t, nfq_max_r, nfq_max_s, nfq_max_sec, nfq_max_rt = np.max(nfq_results, axis=0).T
nfq_min_t, nfq_min_r, nfq_min_s, nfq_min_sec, nfq_min_rt = np.min(nfq_results, axis=0).T
nfq_mean_t, nfq_mean_r, nfq_mean_s, nfq_mean_sec, nfq_mean_rt = np.mean(nfq_results, axis=0).T
nfq_x = np.arange(len(nfq_mean_s))

matplotlib.pyplot.style.use('fivethirtyeight')
params = {
    "figure.figsize": (15, 8), "font.size": 24, "legend.fontsize": 20,
    "axes.titlesize": 28, "axes.labelsize": 24, "xtick.labelsize": 20, "ytick.labelsize": 20
}

matplotlib.pylab.rcParams.update(params)

fig, axs = matplotlib.pyplot.subplots(5, 1, figsize=(15, 30), sharey=False, sharex=True)

axs[0].plot(nfq_max_r, "y", linewidth=1)
axs[0].plot(nfq_min_r, "y", linewidth=1)
axs[0].plot(nfq_mean_r, "y", label="NFQ", linewidth=2)
axs[0].fill_between(nfq_x, nfq_min_r, nfq_max_r, facecolor="y", alpha=0.3)

axs[1].plot(nfq_max_s, "y", linewidth=1)
axs[1].plot(nfq_min_s, "y", linewidth=1)
axs[1].plot(nfq_mean_s, "y", label="NFQ", linewidth=2)
axs[1].fill_between(nfq_x, nfq_min_s, nfq_max_s, facecolor="y", alpha=0.3)

axs[2].plot(nfq_max_t, "y", linewidth=1)
axs[2].plot(nfq_min_t, "y", linewidth=1)
axs[2].plot(nfq_mean_t, "y", label="NFQ", linewidth=2)
axs[2].fill_between(nfq_x, nfq_min_t, nfq_max_t, facecolor="y", alpha=0.3)

axs[3].plot(nfq_max_sec, "y", linewidth=1)
axs[3].plot(nfq_min_sec, "y", linewidth=1)
axs[3].plot(nfq_mean_sec, "y", label="NFQ", linewidth=2)
axs[3].fill_between(nfq_x, nfq_min_sec, nfq_max_sec, facecolor="y", alpha=0.3)

axs[4].plot(nfq_max_rt, "y", linewidth=1)
axs[4].plot(nfq_min_rt, "y", linewidth=1)
axs[4].plot(nfq_mean_rt, "y", label="NFQ", linewidth=2)
axs[4].fill_between(nfq_x, nfq_min_rt, nfq_max_rt, facecolor="y", alpha=0.3)

axs[0].set_title("NFQ: Moving Avg Reward (Training)")
axs[1].set_title("NFQ: Moving Avg Reward (Evaluation)")
axs[2].set_title("NFQ: Total Steps")
axs[3].set_title("NFQ: Training Time")
axs[4].set_title("NFQ: Wall-clock Time")

matplotlib.pyplot.xlabel("Episodes")
axs[0].legend(loc="upper left")
matplotlib.pyplot.show()

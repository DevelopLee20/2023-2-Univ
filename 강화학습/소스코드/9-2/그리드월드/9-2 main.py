import torch.nn as nn               # 파이토치 라이브러리
import torch.nn.functional as F     # 파이토치 라이브러리
import torch                        # 파이토치 라이브러리
import numpy as np                  # 넘파이 라이브러리
from tqdm import tqdm               # 진행도 출력 라이브러리
import matplotlib.pyplot as plt     # 그래프 출력 라이브러리
import matplotlib.pylab as pylab    # 그래프 출력 라이브러리
import time                         # 학습 시간 측정 라이브러리
import random                       # 랜덤 라이브러리
import gym                          # Gym 강화학습 환경 라이브러리

# Fully Connected Deep Discrete-Action Policy, 완전 연결 딥러닝 이산-행동 정책 신경망
class FCDAP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32), activation_fc=F.relu):
        # input_dim: 입력 차원수
        # output_dim: 출력 차원수
        # hidden_dims: 은닉층 차원수 (default: (32, 32))
        # activation_fc: 활성화 함수 (default: F.relu)
        super(FCDAP, self).__init__()       # 기본 클래승의 매개변수를 사용합니다.
        self.activation_fc = activation_fc  # 활성화 함수, 기본값은 relu

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])   # 입력 레이어입니다.
        self.hidden_layers = nn.ModuleList()  # 동적인 은닉층을 생성할 때, 리스트에 넣어 사용합니다.

        for i in range(len(hidden_dims)-1): # 은닉층 차원수만큼 생성합니다.
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1]) # 은닉층 완전연결계층을 생성합니다.
            self.hidden_layers.append(hidden_layer) # 생성된 완전연결계층을 ModuleList에 추가합니다.
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim) # 출력 레이어입니다.

        device = "cpu"                  # 기본 device cpu로 설정
        if torch.cuda.is_available():   # torch의 cuda 설정 가능할 경우
            device = "cuda:0"           # device cuda로 설정
        
        self.device = torch.device(device)  # 설정된 device로 학습
        self.to(self.device)          # 모듈의 파라미터/버퍼 등을 해당 디바이스로 이동 (casting)

    def _format(self, state):   # 입력값이 텐서가 아니면 텐서로 변환
        x = state

        if not isinstance(x, torch.Tensor): # 입력값의 인스턴스가 파이토치 텐서인지 비교
            x = torch.tensor(x, device=self.device, dtype=torch.float32)    # 아닐 경우 토치형식으로 새로 생성
            x = x.unsqueeze(0) # 텐서의 차원을 하나 증가
        
        return x
    
    def forward(self, state):   # 신경망 순전파 함수
        x = self._format(state)                     # 토치 텐서로 변환
        x = self.activation_fc(self.input_layer(x))   # 활성화 함수 설정

        for hidden_layer in self.hidden_layers:     # 은닉층의 수에 따라 반복
            x = self.activation_fc(hidden_layer(x))   # 활성화 함수 지정
        
        x = self.output_layer(x)                    # 출력 레이어 설정

        return x
    
    def full_pass(self, state): # 순전파 수행 후 행동 정책과 로그 계산
        logits = self.forward(state)                            # 순전파 함수 호출
        # 행동의 카테고리형(일반적인 데이터) 확률 분포 생성 - 샘플링
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()  # 생성된 분포 바탕 샘플링

        logpa = dist.log_prob(action).unsqueeze(-1) # 지정된 행동 정책에 대한 로그 값

        is_exploratory = action != np.argmax(logits.detach().numpy())   # 탐색 여부 저장

        return action.item(), is_exploratory.item(), logpa # 행동과 탐색 여부, 로그 값 반환
    
    def select_action(self, state): # 행동 선택 메소드
        logits = self.forward(state)    # 순전파 진행 후 저장
        # 순전파를 바탕으로 행동 카테고리형 확률 분포 생성
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()  # 생성된 분포 바탕 샘플링 진행

        return action.item()    # 행동값 반환

    def select_greedy_action(self, state): # 그리디 행동 정책 선택
        logits = self.forward(state)        # 순전파 진행 후 저장

        return np.argmax(logits.detach().numpy())   # 가치가 최대인 값을 반환
    

class REINFORCE(): # REINFORCE 에이전트 클래스 
    def __init__(self, policy_model_fn, policy_optimizer_fn, policy_optimizer_lr):
        self.policy_model_fn = policy_model_fn          # 정책 신경망 함수 지정
        self.policy_optimizer_fn = policy_optimizer_fn  # 정책 최적화 함수 지정
        self.policy_optimizer_lr = policy_optimizer_lr  # 정책 최적화 함수 학습률 지정

    # 신경망 학습 모델 메소드
    def optimize_model(self):
        T = len(self.rewards)   # 에피소드의 길이 저장
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)   # 할인율 감가 계산
        # 할인율 감가 적용 보상 계산
        returns = np.array([np.sum(discounts[:T-t] * self.rewards[t:]) for t in range(T)])  
        
        discounts = torch.FloatTensor(discounts).unsqueeze(1)   # 할인율 텐서화 및 차원 추가
        returns = torch.FloatTensor(returns).unsqueeze( 1)      # 반환값 텐서화 및 차원 추가
        self.logpas = torch.cat(self.logpas)                    # 로그 확률 텐서화

        policy_loss = -(discounts * returns * self.logpas).mean()  # 기울기 감가 적용
        policy_loss = -(returns * self.logpas).mean()              # 기울기 감가 적용하지 않음

        self.policy_optimizer.zero_grad()   # 정책 최적화 함수 기울기 초기화
        policy_loss.backward()              # 정책 오차 바탕 역전파 발산
        self.policy_optimizer.step()        # 역전파를 바탕으로 갱신

    # 행동 선택 후 다음 상태로 전이하는 메소드
    def interaction_step(self, state, env):
        action, is_exploratory, logpa = self.policy_model.full_pass(state)  # 정책 모델 바탕 한 행동 선택
        new_state, reward, terminated, truncated, info = env.step(action)   # 행동을 바탕으로 다음 스텝 진행
        done = terminated or truncated                                      # 종료 여부 저장

        self.logpas.append(logpa)                           # 로그 확률 저장
        self.rewards.append(reward)                         # 보상 저장
        self.episode_reward[-1] += reward                   # 보상 계산
        self.episode_timestep[-1] += 1                      # 타임 스텝 계산
        self.episode_exploration[-1] += int(is_exploratory) # 탐색 여부 저장 
        
        return new_state, done # 다음 상태 및 종료 여부 반환
    
    # 신경망 훈련 메소드
    def train(self, env, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')    # 훈련 시작, 디버깅 시간 초기화
        self.seed = seed    # 랜덤 시드 초기화
        self.gamma = gamma  # 할인율 초기화

        torch.manual_seed(self.seed)    # 토치 랜덤 시드 지정
        np.random.seed(self.seed)       # 넘파이 랜덤 시드 지정
        random.seed(self.seed)          # 랜덤 랜덤 시드 지정

        nS, nA = env.observation_space.shape[0], env.action_space.n # 상태/행동 수

        self.episode_timestep = []      # 에피소드 타임 스텝 리스트 초기화
        self.episode_reward = []        # 에피소드 보상 리스트 초기화
        self.episode_seconds = []       # 에피소드 시간 리스트 초기화
        self.episode_exploration = []   # 에피소드 탐색 여부 리스트 초기화
        self.evaluation_scores = []     # 평가 점수 리스트 초기화

        # 정책 모델 함수 지정
        self.policy_model = self.policy_model_fn(nS, nA)
        # 정책 최적화 함수 지정
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, self.policy_optimizer_lr)
        
        result = np.empty((max_episodes, 5))    # 결과 넘파이 리스트 초기화
        result[:] = np.nan                      # 결과 넘파이 값 초기화 (Not a Number)
        training_time = 0                       # 학습 시간 초기화

        # 에피소드 수만큼 반복
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()                     # 에피소드 시간 초기화
            state, done = env.reset(seed=self.seed), False  # 환경 초기화
            state = state[0]                                # 환경 초기화 후 상태 저장
            self.episode_reward.append(0.0)                 # 에피소드 보상 리스트에 공간 할당
            self.episode_timestep.append(0.0)               # 에피소드 타임 스텝 리스트에 공간 할당
            self.episode_exploration.append(0.0)            # 에피소드 탐색 여부 리스트에 공간 할당
            self.logpas, self.rewards = [], []              # 로그 확률, 보상 리스트 초기화

            while not done: # 종료일 때까지 반복
                state, done = self.interaction_step(state, env) # 행동 선택 후 다음 상태로 전이
            
            self.optimize_model()   # 신경망 훈련 진행

            episode_elapsed = time.time() - episode_start   # 경과 시간 계산
            self.episode_seconds.append(episode_elapsed)    # 경과시간 리스트에 추가
            training_time += episode_elapsed                # 훈련 시간 계산
            total_step = int(np.sum(self.episode_timestep)) # 누적 타임 스텝 계산

            # 훈련 종료 후 모델 평가 수행 및 그래프 값 계산
            # evaluation_score, _ = self.evaluate(self.online_model, env)
            evaluation_score, _ = self.evaluate(self.policy_model, env)
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
        
        # final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        final_eval_score, score_std = self.evaluate(self.policy_model, env, n_episodes=100)

        wallclock_time = time.time() - training_start
        print("Training complete.")
        print(f"Final evaluation score {final_eval_score:.2f}\u00B1{score_std:.2f} in {training_time:.2f}s training time, {wallclock_time:.2f}s wall-clock time.\n")
        env.close()
        del env

        return result, final_eval_score, training_time, wallclock_time
    
    # 에피소드/훈련 종료 후 그리디 전략으로 보상 계산
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, greedy=True):
        rs = []

        for _ in range(n_episodes):
            s, done = eval_env.reset(), False
            s = s[0]
            rs.append(0)

            while not done:
                if greedy:  # 그리디 행동 전략 지정
                    a = eval_policy_model.select_greedy_action(s)
                else:       # 랜덤 행동 전략 지정
                    a = eval_policy_model.select_action(s)
                
                s, r, terminated, truncated, info = eval_env.step(a)
                done = terminated or truncated
                rs[-1] += r
        
        return np.mean(rs), np.std(rs)
    
    # 학습 종료 후 그리디 전략으로 렌더링 진행
    def render_after_train(self, r_env, n_episodes=1):
        for _ in range(n_episodes):
            s, done = r_env.reset(), False
            s = s[0]

            while not done:
                r_env.render()
                a = self.policy_model.select_greedy_action(s)
                s, r, terminated, truncated, info = r_env.step(a)
                done = terminated or truncated

# 메인 코드
reinforce_results = []

SEEDS = (12, 34, 56, 78, 90)

# 5개의 시드를 반복해 학습을 진행
for seed in SEEDS:
    environment_settings = {
        "env_name": "CartPole-v1",
        "gamma": 1.00,
        "max_minutes": 20,
        "max_episodes": 10000,
        "goal_mean_100_reward": 475
    }

    policy_model_fn = lambda nS, nA: FCDAP(nS, nA, hidden_dims=(128, 64))
    policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0005

    env_name, gamma, max_minutes, max_episodes, goal_mean_100_reward = environment_settings.values()
    env = gym.make(env_name)

    agent = REINFORCE(policy_model_fn, policy_optimizer_fn, policy_optimizer_lr)
    result, final_eval_score, training_time, wallclock_time = agent.train(env, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    
    reinforce_results.append(result)

reinforce_results = np.array(reinforce_results)
agent.render_after_train(env)

reinforce_max_t, reinforce_max_r, reinforce_max_s, reinforce_max_sec, reinforce_max_rt = np.max(reinforce_results, axis=0).T
reinforce_min_t, reinforce_min_r, reinforce_min_s, reinforce_min_sec, reinforce_min_rt = np.min(reinforce_results, axis=0).T
reinforce_mean_t, reinforce_mean_r, reinforce_mean_s, reinforce_mean_sec, reinforce_mean_rt = np.mean(reinforce_results, axis=0).T
reinforce_x = 2200

# 모든 통계값을 그래프로 출력
plt.style.use("fivetirtyeight")

params = {
    "figure.figsize": (15, 8), "font.size": 24, "legend.fontsize": 20,
    "axes.titlesize": 28, "axes.labelsize": 24, "xtick.labelsize": 20, "ytick.labelsize": 20
}

pylab.rcParams.update(params)

fig, axs = plt.subplots(5, 1, figsize=(15, 30), sharey=False, sharex=True)

axs[0].plot(reinforce_max_r, "y", linewidth=1)
axs[0].plot(reinforce_min_r, "y", linewidth=1)
axs[0].plot(reinforce_mean_r, "y", label="reinforce", linewidth=2)
axs[0].fill_between(reinforce_x, reinforce_min_r, reinforce_max_r, facecolor="y", alpha=0.3)

axs[1].plot(reinforce_max_s, "y", linewidth=1)
axs[1].plot(reinforce_min_s, "y", linewidth=1)
axs[1].plot(reinforce_mean_s, "y", label="reinforce", linewidth=2)
axs[1].fill_between(reinforce_x, reinforce_min_s, reinforce_max_s, facecolor="y", alpha=0.3)

axs[2].plot(reinforce_max_t, "y", linewidth=1)
axs[2].plot(reinforce_min_t, "y", linewidth=1)
axs[2].plot(reinforce_mean_t, "y", label="reinforce", linewidth=2)
axs[2].fill_between(reinforce_x, reinforce_min_t, reinforce_max_t, facecolor="y", alpha=0.3)

axs[3].plot(reinforce_max_sec, "y", linewidth=1)
axs[3].plot(reinforce_min_sec, "y", linewidth=1)
axs[3].plot(reinforce_mean_sec, "y", label="reinforce", linewidth=2)
axs[3].fill_between(reinforce_x, reinforce_min_sec, reinforce_max_sec, facecolor="y", alpha=0.3)

axs[4].plot(reinforce_max_rt, "y", linewidth=1)
axs[4].plot(reinforce_min_rt, "y", linewidth=1)
axs[4].plot(reinforce_mean_rt, "y", label="reinforce", linewidth=2)
axs[4].fill_between(reinforce_x, reinforce_min_rt, reinforce_max_rt, facecolor="y", alpha=0.3)

axs[0].set_title("reinforce: Moving Avg Reward (Training)")
axs[1].set_title("reinforce: Moving Avg Reward (Evaluation)")
axs[2].set_title("reinforce: Total Steps")
axs[3].set_title("reinforce: Training Time")
axs[4].set_title("reinforce: Wall-clock Time")

plt.xlabel("Episodes")
axs[0].legend(loc="upper left")

plt.show()
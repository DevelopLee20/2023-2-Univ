import torch                        # torch 라이브러리
import torch.nn as nn               # 상속받을 module 불러오는 라이브러리
import torch.nn.functional as F     # 활성화 함수를 불러오는 라이브러리
import torch.optim as optim         # 최적화 함수를 불러오는 라이브러리
from GridDS_environment import Env  # 목표지점으로 장애물을 피해 이동하는 환경
import numpy as np                  # numpy 라이브러리
from tqdm import *                  # 진행상황과 소요시간을 그래프로 출력하는 라이브러리
import matplotlib.pyplot as plt     # 그래프 출력 라이브러리
import matplotlib.pylab as pylab    # 그래프 출력 라이브러리

class FCQ(nn.Module):
    '''완전연결 신경망 클래스'''
    def __init__(self, input_dim, output_dim, hidden_dims=(32,32), activate_fc=F.relu):
        '''init 기본 생성자 함수
        input_dim: 입력층 차원
        output_dim: 출력층 차원
        hidden_dims: 은닉층 차원
        activate_fc: 활성화 함수, default: relu 함수
        '''
        super(FCQ, self).__init__()                             # 부모 클래스 FCQ의 기본 생성자 불러오기
        self.activate_fc = activate_fc                          # 활성화 함수 설정 -> relu 함수
        self.input_layer = nn.Linear(input_dim, hidden_dims[0]) # 입력 레이어
        self.hidden_layers = nn.ModuleList()                    # 은닉층 레이어, Module List로 작성

        for i in range(len(hidden_dims)-1):                             # 은닉층 차원에 따라 동적으로 변경
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])  # 은닉층 레이어를 생성
            self.hidden_layers.append(hidden_layer)                     # 은닉층 레이어를 Module List에 추가
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)      # 출력층 지정

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
        x = self.activate_fc(self.input_layer(x))   # 활성화 함수 설정

        for hidden_layer in self.hidden_layers:     # 은닉층의 수에 따라 반복
            x = self.activate_fc(hidden_layer(x))   # 활성화 함수 지정
        
        x = self.output_layer(x)                    # 출력 레이어 설정

        return x

class DeepSarsa():
    '''딥살사 에이전트 클래스'''
    def __init__(self, value_model_fn, value_optmizer_fn, value_optimizer_lr):
        '''
        value_model_fn: 신경망 함수, dtype: lambda
        value_optimizer_fn: 신경망 최적화 함수, dtype: lambda
        value_optmizer_lr: 신경망 학습률
        '''
        self.value_model = value_model_fn
        self.value_optimizer_fn = value_optmizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.epsilon = 1    # 입실론 초기값

    def select_action(self, state): # 입실론 그리디 전략으로 행동 선택
        with torch.no_grad():   # 기울기를 저장하지 않아 학습 또는 검증 속도가 향상된다.
            # 신경망 예측 후 .data.numpy() 넘파이 변환 후, .squeeze() 미니배치 차원 제거
            q_values = self.online_model(state).cpu().detach().data.numpy().squeeze()

        if np.random.random() > self.epsilon:           # 입실론 확률에 따라서 랜덤한 행동을 하도록 수행
            action = np.argmax(q_values)                # 가치가 최대인 행동 선택
        else:
            action = np.random.randint(len(q_values))   # 랜덤한 행동을 선택
        
        return action

    def optimizer_model(self, experiences): # 신경망 학습 모델
        states, actions, rewards, new_states, is_terminals = experiences    # 경험 튜플 읽기
        q_sa = self.online_model(states).max(1)[0]                          # 현재 상태의 Q-함수 값 추정
        next_action = self.select_action(new_states)                        # 다음 행동 선택
        q_next_sa = self.online_model(new_states)[0][next_action]           # 선택된 다음 행동으로 Q-함수 값 추정
        targets_q_s = rewards + self.gamma * q_next_sa * (1 - is_terminals) # 딥살사의 타겟(target) 값
        td_errors = q_sa - targets_q_s                                      # 오차 계산
        value_loss = td_errors.pow(2).mul(0.5).mean()                       # 오차를 기반으로 손실함수 계산, 제곱, 루트, 평균
        self.value_optimizer.zero_grad()                                    # 기울기를 0으로 초기화
        value_loss.backward()                                               # 오차를 바탕으로 역전파 발산
        self.value_optimizer.step()                                         # 최적화 함수 갱신

    def train(self, env, gamma, max_episodes):  # 신경망 훈련 메소드
        nS = 15                     # 상태 개수: 15개
        nA = 4                      # 행동 개수: 4개(상, 하, 좌, 우)
        self.gamma = gamma          # 할인율
        min_epsilon = 0.1           # 입실론 최소값
        epsilon_decay_ratio = 0.9   # 입실론 감가율
        self.episode_reward = []
        self.online_model = self.value_model(nS, nA)
        self.value_optimizer = self.value_optimizer_fn(self.online_model, self.value_optimizer_lr)

        for e in tqdm(range(max_episodes), desc='All Episodes'):    # 각 에피소드 진행시 막대 그래프로 표현
            state, is_termial = env.reset(), False  # 상태와 종료 상태를 초기화
            self.episode_reward.append(0)           # 에피소드의 가치를 0으로 초기화

            while not is_termial:                                       # 종료 상태 도달시까지 반복
                action = self.select_action(state)                      # 행동을 선택
                # new_state, reward, is_termial, _ = env.step(action)   # 오류에 의해 반환값(_) 삭제
                new_state, reward, is_termial = env.step(action)        # 한 단계 진행
                self.episode_reward[-1] += reward                       # 에피소드별 보상 리스트에 보상 계산

                # 경험 튜플의 각 원소를 텐서로 변환 후 미니배치 차원 추가
                s = torch.Tensor(state).unsqueeze(0)
                a = torch.Tensor(ToList(action)).unsqueeze(0)
                r = torch.Tensor(ToList(reward)).unsqueeze(0)
                ns = torch.Tensor(new_state).unsqueeze(0)
                done = torch.Tensor(ToList(is_termial)).unsqueeze(0)
                experiences = (s, a, r, ns, done)
                self.optimizer_model(experiences)   # 신경망 학습 실행
                state = new_state                   # 다음 행동 선택
            
            if (self.epsilon > min_epsilon):        # 입실론 최소값 까지 감가
                self.epsilon *= epsilon_decay_ratio # 입실론 값 감가
            
        return self.episode_reward
    
def ToList(d):  # 입력값을 리스트로 반환
    t = []
    t.append(d)

    return t

environment_settings = {        # 환경 설정 매개변수
    "env_name": "Grid World",   # 환경 이름, Custom
    "gamma" : 0.9,              # 할인율, float 형으로 작성
    "max_episodes" : 500        # 최대 에피소드 수, int 형으로 작성
}

# 신경망 모델 구성, dtype: lambda, (30, ), 은닉계층 차원 설정
value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(30,))
# 신경망 최적화 함수, dtype: lambda, 파라미터 설정 및 학습률 지정
value_optmizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
value_optmizer_lr = 0.001   # 학습률 지정

# 각 매개변수 튜플을 변수로 지정
env_name, gamma, max_episodes = environment_settings.values()
env = Env('DeepSarsa')  # 이름 지정 후 환경 인스턴스 생성

# 에이전트 인스턴스 생성(모델, 최적화함수, 학습률) 지정
agent = DeepSarsa(value_model_fn, value_optmizer_fn, value_optmizer_lr)
# 에이전트 학습 후 가치 목록 반환
reward_track = agent.train(env, gamma, max_episodes)

# 반환된 가치 함수 리스트를 그래프로 출력
plt.style.use("fivethirtyeight")
params = {
    "figure.figsize": (15, 8),
    "font.size": 24,
    "legend.fontsize": 20,
    "axes.titlesize": 28,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
}
pylab.rcParams.update(params)

plt.plot(reward_track)
plt.title("Deep Sarsa: Grid World")
plt.xlabel("episode")
plt.ylabel("rewards")
plt.show()

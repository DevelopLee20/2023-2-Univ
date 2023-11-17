# gym 라이브러리 임포트
import gym

# 카트폴-v1 버전의 환경 인스턴스 생성, render_mode가 human이면, 렌더링하는 팝업창을 띄워줍니다.
env = gym.make("CartPole-v1", render_mode="human")

# 목표 step은 500회로 최대 500회 진행 후 에피소드를 종료합니다.
goal_steps = 500

# 에피소드 횟수 만큼 반복합니다. 해당 코드에서는 10회 입니다.
for e in range(10):
    state = env.reset() # 환경 초기화
    state = state[0]    # 환경 초기화 후 상태 저장

    # 목표 step 만큼 반복합니다. 해당 코드에서는 500회 입니다.
    for t in range(goal_steps):
        action = env.action_space.sample() # 상태 튜플에서 랜덤으로 하나의 행동을 뽑습니다.
        state, reward, terminated, truncated, info = env.step(action) # 하나의 step을 진행합니다.
        done = terminated or truncated  # 500 타임 스텝에 도달했거나, 쓰러졌을 경우 done은 True가 됩니다.
        # state: [카트의 위치, 카트 속도, 폴의 각도, 폴의 각속도]
        # reward: 보상
        # done: 종료여부
        print(state, ",", reward, ",", done, ",", truncated, ",", info)

        if done:
            print(f"Episode finished after {t+1} timesteps")
            break

env.close() # 렌더링 창을 닫습니다.

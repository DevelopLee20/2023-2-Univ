## 상태 가치 함수 출력(수정본) - n_cols * n_cols 에 맞게
def print_state_value_function_edited(V, P, n_cols=7, prec=3, title='상태-가치 함수:'):
    print(title)
    wall_state = [0,1,2,3,4,5,6,13,14,15,16,17,19,20,21,27,28,30,32,33,34,35,37,41,42,43,44,45,46,47,48] # 벽 인덱스
    s_count = 0
    dump_list = []; data_list = []

    # 모든 칸을 출력하도록 수정
    for s in range(n_cols * n_cols):
        if s not in wall_state: # s가 벽이 아닐 경우
            v = V[s_count]
            output_value = np.round(v, prec)
            
            if np.all([done for action in P[s_count].values() for _, _, _, done in action]):
                dump_list.append("")
            else:
                output_value = np.round(v, prec)
                dump_list.append(output_value)
                
            s_count += 1
        else:                           # s가 벽일 경우
            dump_list.append("■■■■■■")  # 벽을 출력한다.
            
        if (s + 1) % n_cols == 0:       # 열을 모두 출력했을 때
            data_list.append(dump_list) # 열을 data_list에 저장 후
            dump_list = []              # 열 dump_list 초기화

    data_list[5][6] = "goal"
    table = tabulate(data_list, tablefmt="grid")
    print(table)

# 상태-가치함수 출력
print_state_value_function_edited(V_sarsa, P, title="살사를 통한 상태 가치 함수")

print()

# 최적화된 상태-가치 함수
print_state_value_function_edited(optimal_V, P, title="최적화된 상태-가치 함수")

print()

# 상태-가치 함수의 오차
print_state_value_function_edited(V_sarsa - optimal_V, P, title="상태-가치 함수 오차")
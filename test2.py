import numpy as np
def find_max_transformation_order(frame_similarities, frame_count, random_point):
    frames = sorted(frame_similarities.keys())
    n = len(frames)

    # DP 테이블 초기화: 전환 횟수를 저장하는 테이블
    dp = np.zeros(n)
    path = [None] * n

    for i in range(n):
        for j in range(i):
            if frames[i] - frames[j] >= random_point:
                for transition in frame_similarities[frames[i]]:
                    if transition[0] in [pair[1] for pair in frame_similarities[frames[j]]]:
                        if dp[i] < dp[j] + 1:
                            dp[i] = dp[j] + 1
                            path[i] = (frames[j], transition[0], transition[1])

    # 최대 전환 횟수 및 해당 경로 추적
    max_transitions = max(dp)
    max_index = np.argmax(dp)

    # 경로 재구성
    optimal_path = []
    while max_index is not None and path[max_index] is not None:
        frame, start_csv, end_csv = frames[max_index], path[max_index][1], path[max_index][2]
        optimal_path.append((frame, start_csv, end_csv))
        next_frame = path[max_index][0]
        max_index = frames.index(next_frame) if next_frame in frames else None

    optimal_path.reverse()

    return optimal_path, int(max_transitions)

# 예시 데이터
frame_similarities = {
    2165: [('sync_bts_fakelove_1.csv', 'sync_bts_fakelove_3.csv'), ('sync_bts_fakelove_3.csv', 'sync_bts_fakelove_1.csv')],
    3320: [('sync_bts_fakelove_4.csv', 'sync_bts_fakelove_1.csv'), ('sync_bts_fakelove_1.csv', 'sync_bts_fakelove_4.csv')],
    3605: [('sync_bts_fakelove_4.csv', 'sync_bts_fakelove_1.csv'), ('sync_bts_fakelove_1.csv', 'sync_bts_fakelove_4.csv')],
    5885: [('sync_bts_fakelove_1.csv', 'sync_bts_fakelove_3.csv'), ('sync_bts_fakelove_3.csv', 'sync_bts_fakelove_1.csv')],
    5845: [('sync_bts_fakelove_5.csv', 'sync_bts_fakelove_2.csv'), ('sync_bts_fakelove_2.csv', 'sync_bts_fakelove_5.csv')],
    6530: [('sync_bts_fakelove_6.csv', 'sync_bts_fakelove_2.csv'), ('sync_bts_fakelove_2.csv', 'sync_bts_fakelove_6.csv')],
    1635: [('sync_bts_fakelove_5.csv', 'sync_bts_fakelove_4.csv'), ('sync_bts_fakelove_4.csv', 'sync_bts_fakelove_5.csv')]
}
frame_count = 7000
random_point = 10

optimal_path, max_transitions = find_max_transformation_order(frame_similarities, frame_count, random_point)
print("최적 경로: ", optimal_path)
print("최대 전환 횟수: ", max_transitions)

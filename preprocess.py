import os
import re
import numpy as np
import librosa
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.signal import resample
from video_utils_act import * 

# Mediapipe의 중요 관절 인덱스
key_joint_indices = [
    11, 12, 13, 14,  # 어깨와 팔꿈치
    15, 16,          # 손목
    23, 24,          # 엉덩이
    25, 26           # 무릎
]

def normalize_minmax_mfcc(data):
    """
    Min-Max 정규화 (0~1).
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return data  # Avoid division by zero
    return (data - min_val) / (max_val - min_val)

def normalize_minmax(data, min_val, max_val):
    feature_min = np.min(data)
    feature_max = np.max(data)

    # Division by zero 방지
    range_val = feature_max - feature_min
    if range_val == 0:
        return np.full_like(data, (min_val + max_val) / 2)  # 범위가 0이면 평균값으로 채움

    # Min-Max 정규화
    return ((data - feature_min) / range_val) * (max_val - min_val) + min_val

def extract_mfcc(y, sr, target_frames, n_mfcc=13):
    """Extract and normalize MFCC from a single audio segment."""
    n_fft = min(2048, len(y))
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode='constant')
    hop_length = max(1, int((len(y) - n_fft) / (target_frames - 1)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc[0, :] = np.log1p(np.abs(mfcc[0, :]))

    resampled_mfcc = resample(mfcc, target_frames, axis=1) # Resample mfcc
    return normalize_minmax_mfcc(resampled_mfcc.T)  # Transpose and normalize

def process_audio_segments(wav_path,target_frames, segment_duration=1/30, n_mfcc=13):
    """Process and normalize audio segments."""
    y, sr = librosa.load(wav_path, sr=22050)
    segment_length = int(segment_duration * sr)
    mfcc_segments = []
    for start in range(0, len(y), segment_length):
        segment = y[start:start + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant')
        mfcc = extract_mfcc(segment, sr, target_frames=target_frames, n_mfcc=n_mfcc)
        mfcc_segments.append(mfcc)
    return np.array(mfcc_segments)

def process_video_frame(frame, expend=(1, 1, 4)):
    """
    실시간 프레임을 받아 관절 데이터를 처리하고 정규화.
    """
    # Mediapipe 등으로 관절 데이터 추출
    video_data = extract_pose_from_frame(frame)  # (99, 4)

    # x, y, z 정규화
    xyz_data = video_data[:, :3]
    normalized_data = xyz_data.copy()
    normalized_data[:, 0] = normalize_minmax(xyz_data[:, 0], 0, 1)  # x 정규화
    normalized_data[:, 1] = normalize_minmax(xyz_data[:, 1], 0, 1)  # y 정규화
    normalized_data[:, 2] = normalize_minmax(xyz_data[:, 2], -1, 1)  # z 정규화

    # 상대 위치 계산
    relative_data = process_key_joint_relationship([normalized_data], key_joint_indices)
    final_data = np.concatenate((relative_data, normalized_data), axis=1)  # (120, 3)

    # 크기 조정
    resized_data = zoom(final_data, expend, order=0)  # (120, 3) → (120, 4)
    return resized_data

def process_video_data(video_path, expend=(1, 1, 4)):  # expend==(세로 배율, 가로 배율)
    """Load and normalize video data."""
    video_data = np.load(video_path)  # Load video as (n, 99, 4)

    # x, y, z 값 추출 (visibility 제외)
    xyz_data = video_data[:, :, :3]  # (n, 99, 3)

    # 정규화 수행
    normalized_data = xyz_data.copy()
    normalized_data[:, :, 0] = normalize_minmax(xyz_data[:, :, 0], 0, 1)  # x 정규화 [0, 1]
    normalized_data[:, :, 1] = normalize_minmax(xyz_data[:, :, 1], 0, 1)  # y 정규화 [0, 1]
    normalized_data[:, :, 2] = normalize_minmax(xyz_data[:, :, 2], -1, 1)  # z 정규화 [-1, 1]

    # 정규화된 데이터에서 관절 1, 2, 3 제거
    # Mediapipe 데이터 구조에서 관절 1, 2, 3은 각 사람의 첫 번째 3개 좌표
    # 따라서 제거할 인덱스는 (1, 2, 3)에서 각 사람의 인덱스를 계산
    processed_chunks = []
    for person_idx in range(3):  # 총 3명의 데이터
        # 한 사람당 33개의 관절
        start_idx = person_idx * 33
        end_idx = start_idx + 33

        # 슬라이싱된 데이터에서 제거할 상대적 인덱스
        indices_to_remove = [1, 2, 3]  # 항상 상대적 인덱스 기준으로
        processed_chunk = np.delete(normalized_data[:, start_idx:end_idx, :], indices_to_remove, axis=1)
        processed_chunks.append(processed_chunk)

    relative_data = process_key_joint_relationship(processed_chunks, key_joint_indices) # (n, 30, 3)
    ppl_data = np.concatenate(processed_chunks, axis=1) # (n, 90, 3)
    final_data = np.concatenate((relative_data, ppl_data), axis = 1) # (n, 120, 3)

    # 크기 조정 (선택적)
    resized_data = zoom(final_data, expend, order=0)  # Resize data

    return resized_data, final_data.shape[1]


def process_key_joint_relationship(data_3ppl, key_joint_indices):
    # 1. 중요 관절 추출
    # 중요 관절만 추출한 데이터 저장
    selected_joints_per_person = []

    for person_data in data_3ppl:  # 각 사람의 데이터에 대해
        selected_joints = person_data[:, key_joint_indices, :]  # 중요 관절만 선택
        selected_joints_per_person.append(selected_joints)

    # 3명의 중요 관절 데이터를 결합
    key_joint_data = np.concatenate(selected_joints_per_person, axis=1)  # (n, len(key_joint_indices) * 3, 3)

    # 2. 사람 간 상대 위치 계산
    n, num_joints, _ = key_joint_data.shape  # (n, 30, 3)
    joints_per_person = num_joints // 3  # 한 사람당 관절 수 (10개)

    # 사람별 데이터 분리
    person1 = key_joint_data[:, :joints_per_person, :]  # (n, 10, 3)
    person2 = key_joint_data[:, joints_per_person:2*joints_per_person, :]  # (n, 10, 3)
    person3 = key_joint_data[:, 2*joints_per_person:, :]  # (n, 10, 3)

    # 동일 관절 간 상대 위치 계산
    relative_positions = []
    for person_A, person_B in [(person1, person2), (person1, person3), (person2, person3)]:
        relative_pos = person_B - person_A  # (n, 10, 3)
        relative_positions.append(relative_pos)

    # 리스트를 배열로 변환
    relative_positions_combined = np.concatenate(relative_positions, axis=1)  # (n, 30, 3)

    # 정규화
    relative_positions_normalized = relative_positions_combined.copy()
    relative_positions_normalized[:, :, 0] = normalize_minmax(relative_positions_combined[:, :, 0], 0, 1)  # x 정규화
    relative_positions_normalized[:, :, 1] = normalize_minmax(relative_positions_combined[:, :, 1], 0, 1)  # y 정규화
    relative_positions_normalized[:, :, 2] = normalize_minmax(relative_positions_combined[:, :, 2], -1, 1)  # z 정규화

    return relative_positions_normalized

def repeat_audio_to_match_video(audio_file_paths, target_length,target_frames):
    """
    필요한 음성 세그먼트를 순차적으로 로드하여 비디오 프레임 길이에 맞추고,
    초과된 음성 데이터를 버림.
    """
    combined_segments = []
    audio_idx = 0  # 현재 로드 중인 음성 파일의 인덱스

    while len(combined_segments) < target_length:
        if audio_idx >= len(audio_file_paths):
            audio_idx = 0

        # 현재 음성 파일을 처리
        audio_path = audio_file_paths[audio_idx]
        audio_segments = process_audio_segments(audio_path,target_frames=target_frames)

        # 남은 길이와 현재 음성 세그먼트의 길이를 비교
        remaining_length = target_length - len(combined_segments)
        if len(audio_segments) <= remaining_length:
            # 전체 음성 데이터를 추가하고 다음 음성 파일로 이동
            combined_segments.extend(audio_segments)
            audio_idx += 1
        else:
            # 필요한 만큼만 추가하고 나머지는 버림
            combined_segments.extend(audio_segments[:remaining_length])

    # 정확히 target_length에 맞춰 반환
    return np.array(combined_segments[:target_length])

def combine_audio_video_by_video(expend=(1, 1, 4)):
    """
    각 비디오 파일에 대해 필요한 음성을 순차적으로 로드하며 처리.
    """
    # 수동 경로 설정
    audio_paths = {
        '1': r'/content/drive/MyDrive/SilverAvocado/violence_comb/0/audio',
    }
    video_paths = {
        '1': r'/content/drive/MyDrive/SilverAvocado/violence_comb/0/video',
    }
    comb_paths = {
        '1': r'/content/drive/MyDrive/SilverAvocado/violence_comb/0/comb',
    }
    for label in ['1']:
        audio_label_dir = audio_paths[label]
        video_label_dir = video_paths[label]
        comb_label_dir = comb_paths[label]

        os.makedirs(comb_label_dir, exist_ok=True)

        # 현재 레이블의 모든 음성 파일 로드
        exclude_files = ['.DS_Store']
        audio_files = [
            os.path.join(audio_label_dir, f)
            for f in os.listdir(audio_label_dir)
            if f.endswith(".wav") and f not in exclude_files
        ]

        video_files = [
            os.path.join(video_label_dir, f)
            for f in os.listdir(video_label_dir)
            if f.endswith(".npy") and f not in exclude_files
        ]

        audio_idx = 0  # 각 레이블에서 첫 번째 음성 파일부터 시작

        for video_path in tqdm(video_files, desc=f"Processing Video for Label {label}"):
            print(video_path)
            # 비디오 데이터 로드 및 처리
            video_data,TARGET_FRAMES = process_video_data(video_path, expend=expend)
            video_length = len(video_data)

            # 현재 비디오에 필요한 음성 데이터 로드 및 반복
            repeated_audio_segments = repeat_audio_to_match_video(audio_files[audio_idx:], video_length,TARGET_FRAMES)

            # 비디오와 음성을 결합
            combined_data = []
            for i in range(video_length):
                combined_frame = np.concatenate((repeated_audio_segments[i], video_data[i]), axis=-1)
                combined_data.append(combined_frame)

            combined_data = np.array(combined_data)

            # 결합 데이터를 저장
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_new_name = re.sub(r'_\d+$', '', video_name)
            output_name = f"{video_new_name}_combined.npy"
            output_path = os.path.join(comb_label_dir, output_name)
            np.save(output_path, combined_data)
            print(f"Saved: {output_path}, shape: {combined_data.shape}")


# 실행
combine_audio_video_by_video(expend=(1,1,4)) # (배치크기(고정),세로 배율(고정),가로배율(4 추천))

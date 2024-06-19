import moviepy.editor as mp
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import noisereduce as nr

def compute_spectral_features(y, sr):
    S = np.abs(librosa.stft(y))
    spectral_centroids = librosa.feature.spectral_centroid(S=S, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(S=S, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), sr=sr)
    return spectral_centroids, spectral_bandwidth, chroma_stft, mfcc, S

def compare_spectral_features(features1, features2):
    min_length = min(features1[0].shape[1], features2[0].shape[1])
    centroid_diff = np.mean(np.abs(features1[0][:, :min_length] - features2[0][:, :min_length]))
    bandwidth_diff = np.mean(np.abs(features1[1][:, :min_length] - features2[1][:, :min_length]))
    chroma_diff = np.mean(np.abs(features1[2][:, :min_length] - features2[2][:, :min_length]))
    mfcc_diff = np.mean(np.abs(features1[3][:, :min_length] - features2[3][:, :min_length]))
    return centroid_diff + bandwidth_diff + chroma_diff + mfcc_diff

def find_music_start(y, sr, reference_features, max_trim_seconds=50):
    max_trim_samples = sr * max_trim_seconds
    hop_length = 512
    best_start_time = 0
    best_similarity = float('inf')

    for start_sample in range(0, int(max_trim_samples), hop_length):
        end_sample = start_sample + len(reference_features[0][0])
        if len(y[start_sample:end_sample]) < len(reference_features[0][0]):
            continue

        segment_features = compute_spectral_features(y[start_sample:end_sample], sr)
        similarity = compare_spectral_features(segment_features, reference_features)
        

        if similarity < best_similarity:
            best_similarity = similarity
            best_start_time = start_sample / sr

    return best_start_time

def synchronize_videos(video_files, reference_file, reference_duration=150, max_trim_seconds=50):
    audio_files = []
    for video_file in video_files:
        video = mp.VideoFileClip(video_file)
        audio_file = f"temp_audio_{os.path.basename(video_file)}.wav"
        video.audio.write_audiofile(audio_file, codec='pcm_s16le')
        audio_files.append(audio_file)

    # Extract audio from the reference video file
    reference_video = mp.VideoFileClip(reference_file)
    reference_audio_file = f"temp_audio_{os.path.basename(reference_file)}.wav"
    reference_video.audio.write_audiofile(reference_audio_file, codec='pcm_s16le')

    reference_audio, reference_sr = librosa.load(reference_audio_file, duration=reference_duration)
    reference_audio = nr.reduce_noise(y=reference_audio, sr=reference_sr, freq_mask_smooth_hz=512, prop_decrease=0.3)
    reference_audio = np.nan_to_num(reference_audio)
    reference_features = compute_spectral_features(reference_audio, reference_sr)

    music_start_times = {}
    S_list = []
    sr_list = []

    for audio_file in audio_files:
        y, sr = librosa.load(audio_file)
        y = nr.reduce_noise(y=y, sr=sr, freq_mask_smooth_hz=512, prop_decrease=0.3)
        y = np.nan_to_num(y)  # Replace non-finite values with zeros
        sr_list.append(sr)
        S = compute_spectral_features(y, sr)[4]
        S_list.append(S)

        start_time = find_music_start(y, sr, reference_features, max_trim_seconds)
        music_start_times[audio_file] = start_time

    spectrograms = {}
    for i, video_file in enumerate(video_files):
        audio_file = f"temp_audio_{os.path.basename(video_file)}.wav"
        video = mp.VideoFileClip(video_file)
        start_time = music_start_times[audio_file]

        # trimmed_video = video.subclip(start_time)
        # output_file = f"trimmed_{os.path.basename(video_file)}"
        # trimmed_video.write_videofile(output_file, codec='libx264')

        os.remove(audio_file)

        # Plot and save spectrogram
        plot_title = f"Spectrogram of {video_file}"
        plot_spectrogram(S_list[i], sr_list[i], plot_title)
        spectrogram_image = f"spectrogram_{os.path.basename(video_file)}.png"
        plt.savefig(spectrogram_image)
        plt.close()

        spectrograms[video_file] = spectrogram_image

    # os.remove(reference_audio_file)

    return {video_file: music_start_times[f"temp_audio_{os.path.basename(video_file)}.wav"] for video_file in video_files}, spectrograms

def plot_spectrogram(S, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()

# Example usage
video_files = ["ive_baddie_1.mp4", "ive_baddie_2.mp4", "ive_baddie_3.mp4", "ive_baddie_4.mp4", "ive_baddie_5.mp4", "ive_baddie_6.mp4", "ive_baddie_7.mp4"]
reference_file = "ive_baddie_7.mp4"
starting_times, spectrograms = synchronize_videos(video_files, reference_file)
print("Starting times:", starting_times)
print("Spectrograms:", spectrograms)

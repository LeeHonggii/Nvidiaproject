import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
import json
import random
import time
from adjust import get_adjusted_clip

begin_time = time.time()

dissolve = 0.0

with open("data/output.json", "r", encoding="utf-8") as file:
    json_string_from_file = file.read()

parameter = json.loads(json_string_from_file)

num_stream = parameter["meta_info"]["num_stream"]
# frame_rate = parameter["meta_info"]["frame_rate"]
init_time = parameter["meta_info"]["init_time"]
total_duration = parameter["meta_info"]["duration"]
current_stream = parameter["meta_info"]["first_stream"]
num_cross = parameter["meta_info"]["num_cross"]
folder_path = parameter["meta_info"]["folder_path"]
cross_list = parameter["cross_points"]
sclip_list = []
tclip_list = []

streams = parameter["streams"]
if num_stream != len(streams):
    print(f"ALERT!!! invalid stream number. num_stream: {num_stream} vs len(parameter[\"streams\"]): {len(streams)}")


def load_video_clip(filename):
    try:
        clip = VideoFileClip(filename)
        print(filename)
        return clip
    except FileNotFoundError:
        print(f"FileNotFoundError: {filename}")
    except OSError as e:
        print(f"OSError: {e}")
    return None


for stream in streams:
    filename = folder_path + stream["file"]
    clip = load_video_clip(filename)
    if clip:
        sclip_list.append(clip)
        # if round(clip.fps, 2) != frame_rate:
        #     print(f"ALERT!!! invalide frame rate in file \"{filename}\" {clip.fps} vs parameter[\"frame_rate\"]: {frame_rate}")
        # else:
        #     print(filename)

        if stream["start"] != 0 or stream["end"] != 0:
            sclip_list[-1].subclip(stream["start"], stream["end"])
            print("ALERT!!! we do not support subclip so far")


if num_stream != len(sclip_list):
    print(f"ALERT!!! invalid stream number. num_stream: {num_stream} vs len(sclip_list): {len(sclip_list)}")
    num_stream = len(sclip_list)    # can we continue?

frame_rate = sclip_list[0].fps
for i in range(num_stream):
    if i == 0:
        frame_rate = sclip_list[i].fps
    elif frame_rate != sclip_list[i].fps:
        print(f"ALERT!!! different frame stream 0: {frame_rate} vs stream {i}: sclip_list[i].fps")
        quit()

if "metric" not in parameter["meta_info"] or parameter["meta_info"]["metric"] == "frame":
    frame_gap = 1 / frame_rate
    for cross in cross_list:
        cross["time"] = cross["frame_id"] * frame_gap

last_cross = {
    "time": init_time + total_duration,
    "next_stream": cross_list[-1]["next_stream"],
    "vector_pairs": [
        {
            "vector1": [0, 0, 0, 0],
            "vector2": [0, 0, 0, 0]
        }
    ]
}
cross_list.append(last_cross)
if num_cross + 1 != len(cross_list):
    print("!!!INVLIDS cross number:", num_cross)
else:
    num_cross += 1
print(cross_list)

start_time = init_time
# move_x = 0
# move_y = 0
current_vector = [0, 0, 0, 0]
next_vector = [0, 0, 0, 0]

for i in range(num_cross):
    current_time = cross_list[i]["time_stamp"]
    clip = sclip_list[current_stream].subclip(start_time, current_time)
    clip = get_adjusted_clip(clip, current_vector, next_vector)
    tclip_list.append(clip)
    print(f"stream: {current_stream}, start: {start_time:.1f}, end: {current_time}, cv: {current_vector}, nv: {next_vector}")

    start_time = current_time
    current_stream = cross_list[i]["next_stream"]
    current_vector = cross_list[i]["vector_pairs"][0]["vector1"]
    next_vector = cross_list[i] ["vector_pairs"][0]["vector2"]

    # cx, cy, cw, ch = current_vector
    # ix, iy, iw, ih = next_vector
    #
    # if move_x == 0 and move_y == 0:
    #     tclip_list.append(sclip_list[current_stream].subclip(start_time, current_time))
    # else:
    #     print(f"move x: {move_x} / move y: {move_y}")
    #     temp_clip = sclip_list[current_stream].subclip(start_time, current_time)
    #     moved_clip = temp_clip.set_position((move_x, move_y))
    #     composite_clip = CompositeVideoClip([moved_clip])
    #     # composite_clip = CompositeVideoClip([temp_clip, moved_clip])
    #     tclip_list.append(composite_clip)

    # current_stream = next_stream
    # start_time = current_time
    # # print(f"current_rect: ({cx}, {cy}, {cw}, {ch}) / next_rect: ({ix}, {iy}, {iw}, {ih})")
    # # print(f"diff w: {cw-iw} / diff w%:{abs(cw-iw)/min(cw,iw)/100:.4f}%")
    # # print(f"diff h: {ch-ih} / diff w%:{abs(ch-ih)/min(ch,ih)/100:.4f}%")
    # # print(f"move x: {cx-ix} / move y: {cy-iy}")
    # move_x = cx-ix
    # move_y = cy-iy
    # # move_x = cx-ix + move_x
    # # move_y = cy-iy + move_y


# tclip_list.append(sclip_list[current_stream].subclip(start_time, init_time + total_duration))

print(len(tclip_list))
for i, clip in enumerate(tclip_list):
    print(f"Clip {i} duration: {clip.duration}")

concatenated_clip = concatenate_videoclips(tclip_list)
if total_duration == int(sclip_list[0].duration):
    final_video = concatenated_clip.set_audio(sclip_list[0].audio)
else:
    final_video = concatenated_clip.set_audio(sclip_list[0].subclip(0, total_duration).audio)

output_path = f'data/simple_video_{random.randint(0, 999):03d}.mp4'
final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

for i in range(len(sclip_list)):
    sclip_list[i].close()
concatenated_clip.close()
final_video.close()

end_time = time.time()
elapsed_time = end_time - begin_time
minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60
print(f"Execution time: {minutes:d} minutes and {seconds:.2f} seconds")
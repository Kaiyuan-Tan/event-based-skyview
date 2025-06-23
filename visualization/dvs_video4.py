import cv2
import pandas as pd
import numpy as np
import show_bbox_api

# Paths to the CSV files containing event data and bounding boxes
data_file = 'output/dvs_output.csv'
label_file = 'output/bbox.csv'

# Output video path
video_filename = 'output_video_dvs.avi'

# Frame rate and dimensions
frame_rate = 10
H, W = 720, 1280

# Video writer initialization
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_filename, fourcc, frame_rate, (W, H))

# Read bbox using your custom API
bbox = show_bbox_api.rvt_txt(label_file)  # Must return list of (class_id, (x1, y1, x2, y2), t)
chunk = []
j = 0
bboxes = []

# Read DVS in chunks
chunks = pd.read_csv(data_file, chunksize=10000000)

for df in chunks:
    data = df.to_numpy()
    start_frame = data[0][2]
    end_frame = start_frame + 33300

    for i, row in enumerate(data, start=1):
        chunk.append(row)
        t_current = int(row[2])

        # 获取当前时间戳 t 对应的 bbox（多个）
        if j < len(bbox) and int(bbox[j][2]) == t_current:
            start = j
            while j < len(bbox) and int(bbox[j][2]) == t_current:
                j += 1
            end = j
            bboxes = bbox[start:end]

        # 判断是否该生成一帧
        if (end_frame - row[2]) < 2:
            end_frame = row[2] + 33300

            dvs_img = np.zeros((H, W, 3), dtype=np.uint8)
            chunk_np = np.array(chunk)
            x_coords = chunk_np[:, 0].astype(int)
            y_coords = chunk_np[:, 1].astype(int)
            polarity = chunk_np[:, 3].astype(bool)

            # 坐标越界处理
            mask = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
            x_coords = x_coords[mask]
            y_coords = y_coords[mask]
            polarity = polarity[mask]

            # Red channel for positive polarity (True), Blue for negative (False)
            dvs_img[y_coords[polarity], x_coords[polarity], 2] = 255  # Red
            dvs_img[y_coords[~polarity], x_coords[~polarity], 0] = 255  # Blue

            # Draw bbox
            for name, (xmin, ymin, xmax, ymax), ts in bboxes:
                x1, y1, x2, y2 = map(int, (xmin, ymin, xmax, ymax))
                if str(name) == "0":
                    cv2.rectangle(dvs_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                elif str(name) == "2":
                    cv2.rectangle(dvs_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # 写入视频帧
            video.write(dvs_img)
            chunk = []

    # 处理 chunk 末尾剩下的帧
    if chunk:
        dvs_img = np.zeros((H, W, 3), dtype=np.uint8)
        chunk_np = np.array(chunk)
        x_coords = chunk_np[:, 0].astype(int)
        y_coords = chunk_np[:, 1].astype(int)
        polarity = chunk_np[:, 3].astype(bool)

        mask = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        polarity = polarity[mask]

        dvs_img[y_coords[polarity], x_coords[polarity], 2] = 255
        dvs_img[y_coords[~polarity], x_coords[~polarity], 0] = 255

        # Draw bboxes if match last timestamp
        for name, (xmin, ymin, xmax, ymax), ts in bboxes:
            x1, y1, x2, y2 = map(int, (xmin, ymin, xmax, ymax))
            if str(name) == "0":
                cv2.rectangle(dvs_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            elif str(name) == "2":
                cv2.rectangle(dvs_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        video.write(dvs_img)

# Finalize
video.release()
cv2.destroyAllWindows()
print(f"✅ Video saved as {video_filename}")

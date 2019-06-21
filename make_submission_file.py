import os
from PIL import Image
import numpy as np
import pandas as pd


root_dir = 'test_predicted/'
save_dir = 'submission/'

seq_list = ['sequence%03d' % i for i in range(500)]

print(seq_list)

columnNames = ['pixel_value']

total_result = np.zeros([500 * 10 * 64 * 64 * 3])
total_idx = 0
frame_index = []
for l_idx, l in enumerate(seq_list):
    now_dir = os.path.join(root_dir, l)
    frame_list = [os.path.join(now_dir, f) for f in sorted(os.listdir(now_dir))]
    if len(frame_list) != 10:
        print("Check your output")
        break
    else:
        for f_idx, f in enumerate(frame_list):
            rawData = np.array(Image.open(f))
            frame_index.append(l + '_' + f.split('/')[-1])
            rawData = rawData.flatten()
            total_result[total_idx * 64 * 64 * 3:(total_idx + 1) * 64 * 64 * 3] = rawData

            total_idx += 1

train_data = pd.DataFrame(data=total_result, columns=columnNames)
train_data.insert(loc=0, column='pixel_index', value=range(500 * 10 * 64 * 64 * 3))
train_data.to_csv(os.path.join(save_dir, "submission.csv"), index=False)
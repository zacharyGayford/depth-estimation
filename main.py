import cv2
import torch
import time
import numpy as np
from scipy.interpolate import RectBivariateSpline

model_type = "DPT_Large"
#model_type = "DPT_Hybrid"
#model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
	transform = midas_transforms.dpt_transform
else:
	transform = midas_transforms.small_transform


depth_scale = 0.5
def to_distance(depth, depth_scale):
	return 1.0/(depth*depth_scale)

alpha = 0.2
previous_value = 0.0
def ema_filter(value):
	global preivous_value
	filtered_value = alpha * value + (1 - alpha) * previous_value
	preivous_value = filtered_value
	return filtered_value

video = cv2.VideoCapture(0)
while video.isOpened():

	start = time.time()

	_, original_img = video.read()
	img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

	input = transform(img).to(device)

	with torch.no_grad():

		prediction = midas(input)
		prediction = torch.nn.functional.interpolate(
			prediction.unsqueeze(1),
			size=img.shape[:2],
			mode="bicubic",
			align_corners=False
		).squeeze()

	depth_map = prediction.cpu().numpy()
	depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

	height, width = depth_map.shape
	x_grid = np.arange(width)
	y_grid = np.arange(height)

	spline = RectBivariateSpline(y_grid, x_grid, depth_map)

	distance = to_distance(spline(height / 2, width / 2), depth_scale)

	depth_map = (depth_map * 255).astype(np.uint8)
	depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

	end = time.time()
	frame_time = end - start
	fps = 1 / frame_time

	cv2.putText(original_img, f"fps: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	cv2.putText(depth_map, f"dist: {int(distance)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	cv2.imshow("image", original_img)
	cv2.imshow("depth map", depth_map)

	if cv2.waitKey(5) & 0xFF == 27:
		break

video.release()

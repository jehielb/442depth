import matplotlib
import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize_depth(depth_map, i, colormap='viridis'):
    #  normalized_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_map = np.abs(depth_map.cpu().numpy()[i,:,:])
    # Convert depth map to NumPy array if not already
    depth_map = np.array(depth_map)

    # Normalize depth values to [0, 1]
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    
    if min_val != max_val:
        normalized_depth = (depth_map - min_val) / (max_val - min_val)
    else:
        # Avoid division by zero
        normalized_depth = np.zeros_like(depth_map, dtype=np.float32)

    # Choose colormap
    cmap = plt.get_cmap(colormap)

    # Map normalized depth values to colors
    colored_depth_map = cmap(normalized_depth)

    # Convert to uint8 RGB image
    colored_depth_map_rgb = (colored_depth_map * 255).astype(np.uint8)

    # output_img = Image.fromarray(colored_depth_map_rgb)
    # Save the image to a file (optional)
    # output_img.save('depth_map.png')

    return colored_depth_map_rgb.transpose((2,0,1))



# def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
#     value = value.cpu().numpy()[0,:,:]

#     # normalize
#     vmin = value.min() if vmin is None else vmin
#     vmax = value.max() if vmax is None else vmax
#     if vmin!=vmax:
#         value = (value - vmin) / (vmax - vmin) # vmin..vmax
#     else:
#         # Avoid 0-division
#         value = value*0.
#     # squeeze last dim if it exists
#     #value = value.squeeze(axis=0)

#     cmapper = matplotlib.cm.get_cmap(cmap)
#     value = cmapper(value,bytes=True) # (nxmx4)

#     img = value[:,:,:3]

#     return img.transpose((2,0,1))


# def colorize_depth(depth_map, colormap='viridis'):
    
#     if depth_map.size == 0 or np.isnan(depth_map).any() or np.isinf(depth_map).any():
#         print("Invalid depth map")
#         return None

#     # Normalize depth values to [0, 1]
#     normalized_depth = (depth_map - np.nanmin(depth_map)) / (np.nanmax(depth_map) - np.nanmin(depth_map))


#     # Choose colormap
#     cmap = plt.get_cmap(colormap) # matplotlib.pyplot

#     # Map normalized depth values to colors
#     colored_depth_map = cmap(normalized_depth)

#     # Convert to uint8 RGB image
#     colored_depth_map_rgb = (colored_depth_map[:, :, :3] * 255).astype(np.uint8)

#     output_img = Image.fromarray(colored_depth_map_rgb)
#     # Save the image to a file
#     output_img.save('depth_map.png')

#     return colored_depth_map_rgb
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib import transforms
import cv2
from PIL import Image

from . arrays import is_true


def stack_images_horizontally(file_names, save_file=None):
    ''' Opens the images corresponding to file_names and
    creates a new image stacking them horizontally.
    '''
    images = map(Image.open, file_names)
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGBA', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_file is not None:
        new_im.save(save_file)
    return new_im


def stack_images_in_square_grid(file_names, save_file=None):
    ''' Opens the images corresponding to file_names and
    creates a new grid-square image that plots them in individual cells.
    The behavior is as expected when the sizes of the images are the same.
    '''
    images = map(Image.open, file_names)
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    n_images = len(images)
    im_per_row = int(np.floor(np.sqrt(n_images)))
    total_width = im_per_row * max_width
    total_height = im_per_row * max_height
    new_im = Image.new('RGBA', (total_width, total_height))

    x_offset = 0
    y_offset = 0
    in_row = 0

    for im in images:
        if in_row == im_per_row:
            y_offset += im.size[1]
            x_offset = 0
            in_row = 0

        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
        in_row += 1

    y_offset += im.size[1]

    if save_file is not None:
        new_im.save(save_file)
    return new_im


def read_transparent_png(filename):
    ''' TODO: add docstring
    SEE https://stackoverflow.com/questions/3803888/opencv-how-to-load-png-images-with-4-channels'''

    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:, :, 3]
    rgb_channels = image_4channel[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def rgb_to_hex_string(r, g, b):
    all_ints = is_true.is_integer(r) and is_true.is_integer(g) and is_true.is_integer(b)
    in_range = np.all(np.array([r, g, b]) <= 255) and np.all(np.array([r, g, b]) >= 0)
    if not all_ints or not in_range:
        raise ValueError('Expects integers in [0, 255]')

    return "#{0:02x}{1:02x}{2:02x}".format(int(r), int(g), int(b))


def scalars_to_colors(float_vals, colormap=cm.get_cmap('jet')):
    mappable = cm.ScalarMappable(cmap=colormap)
    colors = mappable.to_rgba(float_vals)
    return colors


def colored_text(in_text, scores=None, colors=None, figsize=(10, 1), colormap=cm.get_cmap('jet'), **kw):
    """
    Input: in_text: (list) of strings
            scores: same size list/array of floats, if None: colors arguement must be not None.
            colors: if not None, it will be used instead of scores.
    """
    fig = plt.figure(frameon=False, figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    t = plt.gca().transData

    if colors is None:
        colors = scalars_to_colors(scores, colormap)

    for token, col in zip(in_text, colors):
        text = plt.text(0, 0, ' ' + token + ' ', color=col, transform=t, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')
    return fig

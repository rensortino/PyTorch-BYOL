import math

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def


def read_images(
    image_dir, shuffle=True, device="mixed", crop_size=224, reader_name="Reader"
):
    preallocate_width_hint = 5980 if device == "mixed" else 0
    preallocate_height_hint = 6430 if device == "mixed" else 0

    images, labels = fn.readers.file(
        file_root=image_dir, random_shuffle=shuffle, name=reader_name
    )
    images = fn.decoders.image_random_crop(
        images,
        device=device,
        output_type=types.RGB,
        preallocate_width_hint=preallocate_width_hint,
        preallocate_height_hint=preallocate_height_hint,
        random_aspect_ratio=[0.8, 1.25],
        random_area=[0.1, 1.0],
        num_attempts=100,
    )
    images = fn.resize(
        images,
        resize_x=crop_size,
        resize_y=crop_size,
        interp_type=types.INTERP_TRIANGULAR,
    )

    return images, labels


def random_grayscale(images, p=0.2):
    if fn.random.coin_flip(probability=p):
        out = fn.color_space_conversion(
            images, image_type=types.RGB, output_type=types.GRAY
        )
        out = fn.color_space_conversion(
            out, image_type=types.GRAY, output_type=types.RGB
        )  # Restore 3 channels
    else:
        out = images
    return out


def gaussian_blur(images, kernel_size):
    image_size = images.shape()[0]
    kernel_size = math.ceil(int(0.1 * image_size))
    images = fn.gaussian_blur(
        images,
        window_size=(kernel_size, kernel_size),
        sigma=fn.random.uniform(range=[0, 1]),
    )


def color_jitter(images):
    b_factor = fn.random.uniform(range=[max(0, 1 - 0.8), 1 + 0.8])
    c_factor = fn.random.uniform(range=[max(0, 1 - 0.8), 1 + 0.8])
    s_factor = fn.random.uniform(range=[max(0, 1 - 0.8), 1 + 0.8])
    h_factor = fn.random.uniform(range=[-0.2 * 360, 0.2 * 360])
    images = fn.color_twist(
        images,
        brightness=b_factor,
        contrast=c_factor,
        saturation=s_factor,
        hue=h_factor,
    )
    return images


def solarize(images, threshold=0.5):
    if fn.random.coin_flip(probability=0.1):
        inverted_img = types.Constant(255, dtype=types.UINT8) - images
        mask = images >= threshold * 255
        out = mask * inverted_img + (True ^ mask) * images
    else:
        out = images
    return out


def simclr_dali_transforms(images, image_size=224):
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
    if fn.random.coin_flip(probability=0.8):
        images = color_jitter(images)
    images = random_grayscale(images, p=0.2)
    kernel_size = int(math.ceil(0.1 * image_size))
    images = fn.gaussian_blur(
        images,
        window_size=(kernel_size, kernel_size),
        sigma=fn.random.uniform(range=[0, 1]),
    )
    if solarize:
        images = solarize(images, threshold=0.5)

    # Reshape the tensor from HWC to CHW layout
    images = fn.transpose(images, output_layout="CHW", perm=[2, 0, 1])
    images = fn.cast(images, dtype=types.FLOAT)

    return images


@pipeline_def(enable_conditionals=True)
def simclr_dali_pipeline(image_dir, shuffle=True, image_size=224, reader_name="Reader"):
    images, labels = read_images(image_dir, shuffle, reader_name=reader_name)
    view1 = simclr_dali_transforms(images, image_size=image_size)
    view2 = simclr_dali_transforms(images, image_size=image_size)
    return view1, view2, labels

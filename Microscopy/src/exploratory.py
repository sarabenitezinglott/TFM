from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

def openczi(path, channels):
    img = AICSImage(path)
    img_data = img.data
    print(img.dims)
    print(img.shape)

    first_channel_data = img.get_image_data(channels, C=1, S=0, T=0)
    return img, first_channel_data

def totiff(path, img, first_channel_data):
    with OmeTiffWriter(path) as writer:
        writer.save(first_channel_data, pixels_physical_size=img.get_physical_pixel_size(),
                    dimension_order="ZYX")

    channel_names = img.get_channel_names()
    return channel_names


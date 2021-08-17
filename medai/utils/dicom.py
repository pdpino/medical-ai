import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut

def dicom_to_np(dicom, voi_lut=True):
    data = dicom.pixel_array.copy()

    if voi_lut:
        data = apply_voi_lut(data, dicom)

    if dicom.PhotometricInterpretation == 'MONOCHROME1':
        data = np.amax(data) - data

    return data


def arr_to_uint8(data):
    data = data - np.min(data)
    data = np.true_divide(data, np.max(data))
    data = (data * 255).astype(np.uint8)

    return data

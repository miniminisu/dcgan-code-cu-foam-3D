import os
import tifffile
import imageio

root_path = '3D_Cu_Foam'
dir_names = os.listdir(root_path)
print(dir_names)
for dir_name in dir_names:
    dir_path = os.path.join(root_path, dir_name)
    image_names = os.listdir(dir_path)  # 获取当前文件夹下所有图像名

    for image_name in image_names:
        slice_dir = os.path.join(dir_path, image_name.split('.tiff')[0]+'_slice')
        if not os.path.exists(slice_dir):
            os.mkdir(slice_dir)
        image_path = os.path.join(root_path, dir_name, image_name)
        image_tif = tifffile.imread(image_path)
        for i in range(image_tif.shape[0]):
            two_d_img = image_tif[i]
            imageio.imwrite(os.path.join(slice_dir, str(i)+'.jpg'), two_d_img)

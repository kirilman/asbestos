from dataset import AsbestosDataSet

dataset = AsbestosDataSet('../task_asbestos_stone_161220-2021_01_13_12_39_03-segmentation mask 1.1 (1)/JPEGImages/asbestos/stones/161220/',
              '../task_asbestos_stone_161220-2021_01_13_12_39_03-segmentation mask 1.1 (1)/SegmentationClass/asbestos/stones/161220/')

print(dataset.n)

for image in dataset:
    print(type(image['image']), type(image['mask']))
    # print(image['image'].filename, image['mask'].filename)
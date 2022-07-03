using JSON, NPZ, BSON, Images
data = NPZ.npzread("dev/train/train_data.npz")

begin
    images = convert.(N0f8, data["imgs"]/255)
    images = permutedims(images, (1, 4, 2, 3))
    images = colorview.(RGB, [images[i, :, :, :] for i in 1:size(images, 1)])
    images = convert(Vector{Matrix{RGB{N0f8}}}, images)
end

# for (idx, image) in images |> enumerate
#     save("julia/data/images/$idx.png", image)
# end


begin
    x = convert.(Int, data["masks"])
    masks = zeros(Int, size(x)[1:3])

    masks[x[:, :, :, 1] .== 255] .= 1
    masks[x[:, :, :, 2] .== 255] .= 2
    masks = Gray.(masks/255)
    
    #masks = cat(masks, zeros(N0f8, size(masks, 1), 1, size(masks, 3), size(masks, 4)), dims = 2)
    #masks = colorview.(RGB, [masks[i, :, :, :] for i in 1:size(masks, 1)])
    #masks = convert(Vector{Matrix{RGB{N0f8}}}, masks)
end

for (idx, mask) in eachslice(masks, dims = 1) |> enumerate
    save("julia/data/labels/$idx.png", mask)
end

classes = ["background", "inside", "outside", "overlap"]

is_background(color) = color == RGB{N0f8}(0, 0, 0)
is_inside(color) = color == RGB{N0f8}(1, 0, 0)
is_outside(color) = color == RGB{N0f8}(0, 1, 0)
is_overlap(color) = is_inside(color) && is_outside(color)

maskCatArray = Vector{Matrix{Int8}}(undef, length(masks))

for i in 1:length(masks)
    maskCatArray[i] = zeros(Int8, size(masks[i], 1), size(masks[i], 2))
    maskCatArray[i][is_background.(masks[i])] .= 1
    maskCatArray[i][is_inside.(masks[i])] .= 2
    maskCatArray[i][is_outside.(masks[i])] .= 3
    maskCatArray[i][is_overlap.(masks[i])] .= 4
end


BSON.@save "julia/data.bson" images masks maskCatArray classes




# using PyCall, BSON, ProgressMeter

# py"""
# import numpy as np
# import albumentations as A
# import cv2

# class PlateDataset:
#     def __init__(self, npz_path = 'train/train_data.npz', multiply = 1):
#         self.multiply = multiply

#         npzfile = np.load(npz_path)
#         self.imgs = npzfile['imgs']
#         self.boxes = np.clip(npzfile['boxes'], 0.0, 1.0)

#     def __len__(self):
#         return len(self.imgs)*self.multiply

#     def __getitem__(self, idx: slice):
#         idx = int(idx/self.multiply)
#         train_transform = A.Compose([
#             A.HorizontalFlip(p=0.5),
#             A.GaussNoise(p=0.5),
#             A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
#             A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.5, rotate_limit=15, p=1, border_mode=cv2.BORDER_CONSTANT)],
#             bbox_params = A.BboxParams(format="albumentations", label_fields=['class_id'])
#         )
        
#         transformed = train_transform(
#             image=self.imgs[idx],
#             bboxes=self.boxes[idx],
#             class_id=[0, 1]
#             )
        
#         box = np.empty((2, 4), dtype=np.float32)

#         idx = 0
#         for id in transformed['class_id']:
#             box[id] = transformed['bboxes'][idx]
#             idx += 1

#         box = np.concatenate((box[0], box[1]), axis=0)

#         img = transformed['image']/255.0
#         img = img.astype(np.float32)

#         return  img, box

#     def getall(self):
#         imgs = np.empty((300, 300, 3, self.__len__()))
#         boxes = np.empty((8, self.__len__()))

#         for i in range(0, len(self.imgs) - 1):
#             temp = self.__getitem__(i)
#             imgs[:, :, :, i] = temp[0]
#             boxes[:, i] = temp[1]

#         return imgs, boxes

# data = PlateDataset()

# def get():
#     return data.getall()
# """

# @showprogress for i in 1:100
#     imgs, boxes = py"get"();
#     BSON.@save "./../julia/data/data$i.bson" imgs boxes
# end
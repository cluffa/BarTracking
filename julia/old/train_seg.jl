using Flux, CUDA, UnicodePlots, Statistics, BSON, Augmentor, Images, Folds
using Flux: @epochs, params, throttle, DataLoader, update!

begin 
    BSON.@load "data.bson" images masks
    imgs = reinterpret(RGB{Float32}, permutedims(images, (3, 1, 2, 4)))[1, :, :, :];
    temp = reinterpret(RGB{Float32}, cat(permutedims(masks, (3, 1, 2, 4)), zeros(Float32, 1, 300, 300, 931), dims = 1))[1, :, :, :];
    masks = zeros(RGB{Float32}, 116, 116, 931);
    for i in size(masks, 3)
        masks[:, :, i] = imresize(temp[:, :, i], (116, 116));
    end
    images = Nothing;
    temp = Nothing;
end;

nobs = size(imgs, 3)

pl = GaussianBlur(1:2:10, 0.1:0.1:10.0) * NoOp() |>
    Zoom(0.7:0.1:2) |>
    Rotate([(-45:5:45)..., (-10:1:10)...])

function apply_aug()
    aug_imgs, aug_masks = zeros(Float32, 300, 300, 3, nobs), zeros(Float32, 116, 116, 2, nobs)
    Folds.map(enumerate(zip(eachslice(imgs; dims=3), eachslice(masks; dims=3)))) do (i, (img, mask))
        img, mask = augment(img => mask, pl)
        img, mask = imresize(img, (300, 300)), imresize(mask, (116, 116))
        img, mask = channelview.((img, mask))
        mask = mask[1:2, :, :]
        img, mask = permutedims(img, (2, 3, 1)), permutedims(mask, (2, 3, 1))
        img, mask = reshape(img, (300, 300, 3, 1)), reshape(mask, (116, 116, 2, 1))
        aug_imgs[:, :, :, i], aug_masks[:, :, :, i] = img, mask
    end
    return aug_imgs, aug_masks
end

function getNewData(batchsize = 8, shuffle = true, multiplications = 15)
    data = apply_aug()
    if multiplications > 1
        for i in 2:multiplications
            aug_imgs, aug_masks = apply_aug()
            data = cat(data[1], aug_imgs, dims = 4), cat(data[2], aug_masks, dims = 4)
        end
    end

    #data = gpu(data[1]), gpu(data[2])

    DataLoader(data, batchsize = batchsize, shuffle = shuffle);
end

BSON.@load "model.bson" m
#BSON.@load "modeluntrained.bson" m

loss(model) = (x, y) -> Flux.Losses.dice_coeff_loss(model(x), y);

function progress()
    r = rand(1:nobs, 16)
    train_loss = loss(m |> gpu)(DataUtils.imgs[:, :, :, r] |> gpu, DataUtils.masks[:, :, :, r] |> gpu)
    @show train_loss
end

# function rgb_to_3d(img)
#     return reshape(img, (300, 300, 3, 1))
# end

# function augment_data(data, pl = pl)
#     outs = similar.(data)
#     augmentbatch!(outs, data, pl)
#     resized = zeros(Float32, 116, 116, size(outs[2], 3))
#     for i in 1:size(outs[2], 3)
#         #resized[:, :, i] = imresize(resized[:, :, i], (116, 116)) 
#     end
#     #outs[2] = resized
# end
# out = augment_data((imgs[:, :, 1:10], masks[:, :, 1:10]))
# out .|> size
# out .|> typeof

function my_custom_train!(loss, ps, data, opt)
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss
    ps = params(ps)
    for d in data
        d = gpu.(d)
        gs = gradient(ps) do
            training_loss = loss(d...)
            # Code inserted here will be differentiated, unless you need that gradient information
            # it is better to do the work outside this block.
            return training_loss
        end
        # Insert whatever code you want here that needs training_loss, e.g. logging.
        # logging_callback(training_loss)
        # Insert what ever code you want here that needs gradient.
        # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
        update!(opt, ps, gs)
        # Here you might like to check validation set accuracy, and break out to do early stopping.
    end
end

begin
    m = m |> gpu

    opt = ADAM(0.001)
    ps = params(m)
    lossfn = loss(m)

    for epoch in 1:500
	@show epoch
        my_custom_train!(lossfn, ps, getNewData(), opt)
    end
    
    m = m |> cpu
    BSON.@save "model.bson" m
end



# function plot_rand()
#     r = rand(1:nobs);
#     #img = imgs[:, :, :, r];
#     pred = m(imgs[:, :, :, r:r] |> cpu);
#     inside = pred[:, :, 1]
#     outside = pred[:, :, 2]
#     UnicodePlots.heatmap(inside + outside, width = 60, height = 60)
# end

# plot_rand()

#for i in 1:10
#    print(plot_rand(), "\n")
#end

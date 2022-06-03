using Flux, CUDA, UnicodePlots, Statistics
using Flux: @epochs, params, throttle, DataLoader
using BSON: @save, @load

include("DataUtils.jl")
using .DataUtils

imgs = gpu(DataUtils.imgs);
masks = gpu(DataUtils.masks);

include("Model.jl")
using .Model: NewModel

model = NewModel(3, 2) |> gpu;

rand(300, 300, 3, 10) |> gpu |> model |> size

loss(x, y) = Flux.Losses.dice_coeff_loss(model(x), y)

nobs = size(imgs, 4)

function plot_rand()
    r = rand(1:nobs);
    #img = imgs[:, :, :, r];
    pred = model(imgs[:, :, :, r:r]) |> cpu;
    inside = pred[:, :, 1]
    outside = pred[:, :, 2]
    UnicodePlots.heatmap(inside + outside, width = 60, height = 60)
end

function progress()
    r = rand(1:nobs, 10)
    train_loss = loss(imgs[:, :, :, r], masks[:, :, :, r])
    @show train_loss
end

progress()

data = DataLoader((imgs, masks), batchsize=8, shuffle=true);

@epochs 25 Flux.train!(loss, params(model), data, ADAM(0.005), cb = throttle(progress, 10))
@epochs 100 Flux.train!(loss, params(model), data, ADAM(0.001), cb = throttle(progress, 10))
@epochs 50 Flux.train!(loss, params(model), data, ADAM(0.0001), cb = throttle(progress, 10))


@save "model.bson" model

@load "model.bson" model

model |> gpu;

for i in 1:10
    print(plot_rand(), "\n")
end


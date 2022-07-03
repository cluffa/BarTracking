using Flux, CUDA, UnicodePlots, Statistics, BSON
using Flux: @epochs, params, throttle, DataLoader

BSON.@load "data.bson" images boxes

m = Chain(
    BatchNorm(3),
    Conv((3, 3), 3 => 64, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 64 => 128, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 128 => 256, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 256 => 128, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 128 => 64, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 64 => 8, relu),
    Flux.flatten,
    Dense(200 => 8, sigmoid),
)

rand(Float32, 300, 300, 3, 10) |> m

loss_fn(model) = (x, y) -> Flux.Losses.mse(model(x), y)

nobs = size(images, 4)

modelName = "regmodel.bson"

begin
    try
        BSON.@load modelName m
        @info "Loaded model from file"
    catch
        @info "model not found, training from scratch"
    end
    m = m |> gpu
    images = images |> gpu
    boxes = boxes |> gpu

    loss = loss_fn(m)
    function progress()
        r = rand(1:nobs, 10)
        train_loss = loss(images[:, :, :, r], boxes[:, r])
        @show train_loss
    end

    data = DataLoader((images, boxes), batchsize=16, shuffle=true);

    steps = (
        (10, 0.005),
        (250, 0.001),
        (250, 0.0005),
        (100, 0.0001),
        (100, 0.00005),
        (100, 0.00001)
    );

    for (nepochs, lr) in steps
        @info "Training for " * nepochs * " epochs with learning rate " * lr
        @epochs nepochs Flux.train!(loss, params(m), data, ADAM(lr), cb = throttle(progress, 10))
    end

    m = m |> cpu
    images = images |> cpu
    boxes = boxes |> cpu

    BSON.@save modelName m
end

function plot_rand()
    r = rand(1:nobs);
    #img = imgs[:, :, :, r];
    pred = m(imgs[:, :, :, r:r] |> cpu);
    inside = pred[:, :, 1]
    outside = pred[:, :, 2]
    UnicodePlots.heatmap(inside + outside, width = 60, height = 60)
end

plot_rand()

#for i in 1:10
#    print(plot_rand(), "\n")
#end

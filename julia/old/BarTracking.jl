using Flux, BSON, Plots, CUDA

begin
    include("DataUtils.jl")
    using .DataUtils
    imgs = DataUtils.imgs |> gpu
    masks = DataUtils.masks |> gpu
end;

BSON.@load "model.bson" m; m = gpu(m);

m(imgs[:, :, :, 1:10])

function plot_rand()
    r = rand(1:size(imgs, 4));
    pred = m(imgs[:, :, :, r:r]) |> cpu;
    inside = pred[:, :, 1]
    outside = pred[:, :, 2]
    heatmap(inside + outside, width = 60, height = 60)
end

plot_rand()


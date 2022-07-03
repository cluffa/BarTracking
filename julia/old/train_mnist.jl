using Flux, CUDA, Plots, Statistics, BenchmarkTools
using MLDatasets: MNIST
using Flux: onehotbatch, logitcrossentropy, onecold, throttle, params, @epochs, DataLoader, flatten
using BSON: load, @save

CUDA.functional()

images = MNIST().features;
labels = MNIST().targets;

X = reshape(images, 28, 28, 1, 60000) |> gpu;
y = onehotbatch(labels, 0:9) |> gpu
y_smooth = Flux.label_smoothing(y, 0.15f0)

# model = Chain(
#     Dense(28*28, 512, relu),
#     Dense(512, 10),
#     softmax
#     ) |> gpu;

model = Chain(
   # First convolution, operating upon a 28x28 image
   Conv((3, 3), 1=>16, pad=(1,1), relu),
   MaxPool((2,2)),
 
   # Second convolution, operating upon a 14x14 image
   Conv((3, 3), 16=>32, pad=(1,1), relu),
   MaxPool((2,2)),
 
   # Third convolution, operating upon a 7x7 image
   Conv((3, 3), 32=>32, pad=(1,1), relu),
   MaxPool((2,2)),
 
   # Reshape 3d array into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
   flatten,
   Dense(3*3*32, 10)) |> gpu;

loss(X, y) = logitcrossentropy(model(X), y);
#accuracy(X, y) = mean(Flux.onecold(model(X)) .== Flux.onecold(y));
opt = ADAM(0.0001);

progress = () -> @show loss(X, y);

data = DataLoader((X, y_smooth), batchsize=32, shuffle=true);

@epochs 10 Flux.train!(loss, params(model), data, opt, cb = throttle(progress, 10))

begin
    img = X[:, :, :, rand(1:60_000)];
    img = reshape(img, 28, 28, 1, 1);
    pred = model(img)
    img = reshape(img, 28, 28) |> cpu;
    img = permutedims(img, (2, 1))[end:-1:1, :]
    pred = onecold(pred, 0:9) |> cpu
    heatmap(img, title = pred);
end

@save "mnist.bson" model
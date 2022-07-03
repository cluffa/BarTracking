using Metalhead, Flux, BSON, Zygote, MLUtils
@info "packages loaded"
model = Chain(
  Chain([
    Conv((7, 7), 3 => 64, pad=3, stride=2, bias=false),  # 9_408 parameters
    BatchNorm(64, relu),              # 128 parameters, plus 128
    MaxPool((3, 3), pad=1, stride=2),
    Parallel(
      Metalhead.addrelu,
      Chain(
        Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
        BatchNorm(64, relu),          # 128 parameters, plus 128
        Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
        BatchNorm(64),                # 128 parameters, plus 128
      ),
      identity,
    ),
    Parallel(
      Metalhead.addrelu,
      Chain(
        Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
        BatchNorm(64, relu),          # 128 parameters, plus 128
        Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
        BatchNorm(64),                # 128 parameters, plus 128
      ),
      identity,
    ),
    Parallel(
      Metalhead.addrelu,
      Chain(
        Conv((3, 3), 64 => 128, pad=1, stride=2, bias=false),  # 73_728 parameters
        BatchNorm(128, relu),         # 256 parameters, plus 256
        Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
        BatchNorm(128),               # 256 parameters, plus 256
      ),
      Chain([
        Conv((1, 1), 64 => 128, stride=2, bias=false),  # 8_192 parameters
        BatchNorm(128),               # 256 parameters, plus 256
      ]),
    ),
    Parallel(
      Metalhead.addrelu,
      Chain(
        Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
        BatchNorm(128, relu),         # 256 parameters, plus 256
        Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
        BatchNorm(128),               # 256 parameters, plus 256
      ),
      identity,
    ),
    Parallel(
      Metalhead.addrelu,
      Chain(
        Conv((3, 3), 128 => 256, pad=1, stride=2, bias=false),  # 294_912 parameters
        BatchNorm(256, relu),         # 512 parameters, plus 512
        Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
        BatchNorm(256),               # 512 parameters, plus 512
      ),
      Chain([
        Conv((1, 1), 128 => 256, stride=2, bias=false),  # 32_768 parameters
        BatchNorm(256),               # 512 parameters, plus 512
      ]),
    ),
    Parallel(
      Metalhead.addrelu,
      Chain(
        Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
        BatchNorm(256, relu),         # 512 parameters, plus 512
        Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
        BatchNorm(256),               # 512 parameters, plus 512
      ),
      identity,
    ),
    Parallel(
      Metalhead.addrelu,
      Chain(
        Conv((3, 3), 256 => 512, pad=1, stride=2, bias=false),  # 1_179_648 parameters
        BatchNorm(512, relu),         # 1_024 parameters, plus 1_024
        Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
        BatchNorm(512),               # 1_024 parameters, plus 1_024
      ),
      Chain([
        Conv((1, 1), 256 => 512, stride=2, bias=false),  # 131_072 parameters
        BatchNorm(512),               # 1_024 parameters, plus 1_024
      ]),
    ),
    Parallel(
      Metalhead.addrelu,
      Chain(
        Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
        BatchNorm(512, relu),         # 1_024 parameters, plus 1_024
        Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
        BatchNorm(512),               # 1_024 parameters, plus 1_024
      ),
      identity,
    ),
  ]),
  Chain(
    AdaptiveMeanPool((1, 1)),
    MLUtils.flatten,
    Dense(512 => 8),                    # 4_104 parameters
  ),
);
@info "model created"

# Total: 62 trainable arrays, 11_180_616 parameters,
# plus 40 non-trainable, 9_600 parameters, summarysize 42.701 MiB.

function my_custom_train!(loss, ps, data, opt)
    total_loss = 0
    local training_loss
    ps = Flux.params(ps)
    for d in data
        gs = gradient(ps) do
            training_loss = loss(gpu.(d)...)
            return training_loss
        end
        total_loss += training_loss
        Flux.update!(opt, ps, gs)
    end
    @show total_loss
end

begin
    @info "starting training"
    #BSON.@load "boxmodel.bson" model
    model = model |> gpu
    for epoch in 1:100
        BSON.@load "data/data$epoch.bson" imgs boxes
        data = Flux.DataLoader((imgs, boxes), batchsize=4, shuffle=true);
        @info "Epoch #$epoch"
        my_custom_train!((x, y) -> Flux.mse(model(x), y), Flux.params(model), data, Flux.ADAM(0.001))
    end
    @info "saving model"
    model = model |> cpu
    BSON.@save "boxmodel.bson" model
end


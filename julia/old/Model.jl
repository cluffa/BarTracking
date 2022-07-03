using CUDA, Flux, BSON

m = Chain(
    BatchNorm(3),
    Conv((3, 3), 3 => 16, relu),
    BatchNorm(16),
    Conv((3, 3), 16 => 16, relu),
    SkipConnection(
        Chain(
            MaxPool((2, 2)),
            BatchNorm(16),
            Conv((3, 3), 16 => 32, relu),
            BatchNorm(32),
            Conv((3, 3), 32 => 32, relu),
            SkipConnection(
                Chain(
                    MaxPool((2, 2)),
                    BatchNorm(32),
                    Conv((3, 3), 32 => 64, relu),
                    BatchNorm(64),
                    Conv((3, 3), 64 => 64, relu),
                    SkipConnection(
                        Chain(
                            MaxPool((2, 2)),
                            BatchNorm(64),
                            Conv((3, 3), 64 => 128, relu),
                            BatchNorm(128),
                            Conv((3, 3), 128 => 128, relu),
                            SkipConnection(
                                Chain(
                                    MaxPool((2, 2)),
                                    BatchNorm(128),
                                    Conv((3, 3), 128 => 256, relu),
                                    BatchNorm(256),
                                    Conv((3, 3), 256 => 256, relu),
                                    ConvTranspose((2, 2), 256 => 128, stride = 2)
                                ),
                                (mx, x) -> cat(x[(1:size(mx)[1]) .+ ((size(x)[1] - size(mx)[1]) ÷ 2), (1:size(mx)[2]) .+ ((size(x)[2] - size(mx)[2]) ÷ 2), :, :], mx, dims = 3)
                            ),
                            BatchNorm(256),
                            Conv((3, 3), 256 => 128, relu),
                            BatchNorm(128),
                            Conv((3, 3), 128 => 128, relu),
                            ConvTranspose((2, 2), 128 => 64, stride = 2)
                        ),
                        (mx, x) -> cat(x[(1:size(mx)[1]) .+ ((size(x)[1] - size(mx)[1]) ÷ 2), (1:size(mx)[2]) .+ ((size(x)[2] - size(mx)[2]) ÷ 2), :, :], mx, dims = 3)
                    ),
                    BatchNorm(128),
                    Conv((3, 3), 128 => 64, relu),
                    BatchNorm(64),
                    Conv((3, 3), 64 => 64, relu),
                    ConvTranspose((2, 2), 64 => 32, stride = 2),
                ),
                (mx, x) -> cat(x[(1:size(mx)[1]) .+ ((size(x)[1] - size(mx)[1]) ÷ 2), (1:size(mx)[2]) .+ ((size(x)[2] - size(mx)[2]) ÷ 2), :, :], mx, dims = 3)
            ),
            BatchNorm(64),
            Conv((3, 3), 64 => 32, relu),
            BatchNorm(32),
            Conv((3, 3), 32 => 32, relu),
            ConvTranspose((2, 2), 32 => 16, stride = 2),
        ),
        (mx, x) -> cat(x[(1:size(mx)[1]) .+ ((size(x)[1] - size(mx)[1]) ÷ 2), (1:size(mx)[2]) .+ ((size(x)[2] - size(mx)[2]) ÷ 2), :, :], mx, dims = 3)
    ),
    BatchNorm(32),
    Conv((3, 3), 32 => 16, relu),
    BatchNorm(16),
    Conv((3, 3), 16 => 16, relu),
    Conv((1, 1), 16 => 2, sigmoid)
)

BSON.@save "modeluntrained.bson" m
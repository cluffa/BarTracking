module Model
    export NewModel, concat_and_crop
    using Flux
    
    function concat_and_crop(mx::AbstractArray{T,4}, x::AbstractArray{T,4}) where T
        w, h = size(x)
        mw, mh = size(mx)
        rx = (1:mw) .+ ((w - mw) ÷ 2)
        ry = (1:mh) .+ ((h - mh) ÷ 2)
        return cat(x[rx, ry, :, :], mx, dims = 3)
    end
    
    NewModel(in, out) = Chain(
        BatchNorm(in),
        Conv((3, 3), in => 16, relu),
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
                                    concat_and_crop
                                ),
                                BatchNorm(256),
                                Conv((3, 3), 256 => 128, relu),
                                BatchNorm(128),
                                Conv((3, 3), 128 => 128, relu),
                                ConvTranspose((2, 2), 128 => 64, stride = 2)
                            ),
                            concat_and_crop
                        ),
                        BatchNorm(128),
                        Conv((3, 3), 128 => 64, relu),
                        BatchNorm(64),
                        Conv((3, 3), 64 => 64, relu),
                        ConvTranspose((2, 2), 64 => 32, stride = 2),
                    ),
                    concat_and_crop
                ),
                BatchNorm(64),
                Conv((3, 3), 64 => 32, relu),
                BatchNorm(32),
                Conv((3, 3), 32 => 32, relu),
                ConvTranspose((2, 2), 32 => 16, stride = 2),
            ),
            concat_and_crop
        ),
        BatchNorm(32),
        Conv((3, 3), 32 => 16, relu),
        BatchNorm(16),
        Conv((3, 3), 16 => 16, relu),
        Conv((1, 1), 16 => out, sigmoid)
    )
end
module DataUtils
    export imgs, masks
    
    using Flux: DataLoader
    using NPZ: npzread
    
    struct WLImage 
    end



    data = npzread("train_data.npz");

    imgs = data["imgs"] |> (x -> permutedims(Float32.(x)./255, (2, 3, 4, 1)));
    masks = data["masks"] |> (x -> permutedims(Float32.(x)./255, (2, 3, 4, 1)));
end